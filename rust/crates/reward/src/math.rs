//! Math answer verification without a Python CAS.
//!
//! A toy reward compares the last number as a string, which rejects equivalent
//! answers such as `1/2` vs `0.5`. We normalize LaTeX, parse a restricted
//! arithmetic grammar into an AST, and compare expressions numerically at
//! several deterministic sample points. No `eval`, no arbitrary code.

use std::collections::BTreeSet;

/// Pull the final answer out of a chain-of-thought response.
pub fn extract_answer(text: &str) -> String {
    let mut boxed = None;
    let bytes = text.as_bytes();
    let mut search = 0;
    while let Some(rel) = text[search..].find("\\boxed") {
        let start = search + rel + "\\boxed".len();
        let mut i = start;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'{' {
            if let Some(end) = matching_brace(text, i) {
                boxed = Some(text[i + 1..end].to_string());
                search = end + 1;
                continue;
            }
        }
        search = start;
    }
    let candidate = boxed.or_else(|| {
        for marker in [
            "answer is",
            "answer:",
            "the answer is",
            "result is",
            "final answer",
            "答案是",
            "答案为",
        ] {
            if let Some(pos) = text.rfind(marker) {
                let tail = &text[pos + marker.len()..];
                if !tail.trim().is_empty() {
                    return Some(tail.to_string());
                }
            }
        }
        // Plain text such as "answer 42" has no marker: fall back to the last
        // numeric token (including fractions and scientific notation).
        last_number(text).or_else(|| {
            text.lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .map(str::to_string)
        })
    });
    let mut candidate = candidate.unwrap_or_default();

    candidate = candidate.trim().to_string();
    while candidate.ends_with('.') && !candidate.ends_with("..") {
        // Trailing sentence punctuation, not a decimal point.
        candidate.pop();
        candidate = candidate.trim_end().to_string();
    }
    candidate.trim_matches(|c| c == '$' || c == '`').trim().to_string()
}

/// Last numeric token in free-form text, including signed fractions and
/// scientific notation. Used only when there is no boxed/marked answer.
fn last_number(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        if (chars[i] == '-' || chars[i] == '+')
            && i + 1 < chars.len()
            && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.')
        {
            i += 1;
        }
        if i >= chars.len()
            || !(chars[i].is_ascii_digit()
                || (chars[i] == '.' && chars.get(i + 1).map(|c| c.is_ascii_digit()).unwrap_or(false)))
        {
            i += 1;
            continue;
        }
        let mut has_digit = false;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
            has_digit = true;
        }
        if i < chars.len() && chars[i] == '.' {
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
                has_digit = true;
            }
        }
        if has_digit && i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
            let mut j = i + 1;
            if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                j += 1;
            }
            let digits = j;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > digits {
                i = j;
            }
        }
        if has_digit && i < chars.len() && chars[i] == '/' {
            let mut j = i + 1;
            if j < chars.len() && (chars[j] == '-' || chars[j] == '+') {
                j += 1;
            }
            let digits = j;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > digits {
                i = j;
            }
        }
        if has_digit {
            best = Some((start, i));
        } else {
            i = start + 1;
        }
    }
    best.map(|(start, end)| chars[start..end].iter().collect())
}

fn matching_brace(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Normalize LaTeX/unicode math into the restricted grammar the parser reads.
pub fn normalize_answer(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut prev_digit = false;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '$' | '`' | '\n' | '\r' | '\t' | ' ' | '~' => {}
            ',' => {
                // Thousands separator between digits is noise; a separator in
                // an argument list is meaningful.
                let next = chars.get(i + 1).copied();
                if !(prev_digit && next.map(|n| n.is_ascii_digit()).unwrap_or(false)) {
                    out.push(',');
                    prev_digit = false;
                }
            }
            '×' | '⋅' => out.push('*'),
            '÷' => out.push('/'),
            '−' | '–' => out.push('-'),
            'π' => out.push_str("pi"),
            '√' => out.push_str("sqrt"),
            '\\' => {
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                let cmd: String = chars[i + 1..j].iter().collect();
                match cmd.as_str() {
                    "frac" | "dfrac" | "tfrac" | "cfrac" => {
                        if let Some((arg, end)) = braced_group(&chars, j) {
                            if let Some((den, end2)) = braced_group(&chars, end + 1) {
                                out.push('(');
                                out.push_str(&normalize_answer(&arg));
                                out.push_str(")/(");
                                out.push_str(&normalize_answer(&den));
                                out.push(')');
                                i = end2;
                                prev_digit = false;
                                continue;
                            }
                        }
                    }
                    "sqrt" => {
                        let mut k = j;
                        let mut root = None;
                        if k < chars.len() && chars[k] == '[' {
                            if let Some(close) = chars[k..].iter().position(|&x| x == ']') {
                                root = Some(chars[k + 1..k + close].iter().collect::<String>());
                                k += close + 1;
                            }
                        }
                        if let Some((arg, end)) = braced_group(&chars, k) {
                            match root {
                                Some(n) => {
                                    out.push('(');
                                    out.push_str(&normalize_answer(&arg));
                                    out.push_str(")^(1/(");
                                    out.push_str(&n);
                                    out.push_str("))");
                                }
                                None => {
                                    out.push_str("sqrt(");
                                    out.push_str(&normalize_answer(&arg));
                                    out.push(')');
                                }
                            }
                            i = end;
                            prev_digit = false;
                            continue;
                        }
                    }
                    "text" | "mathrm" | "mathbf" | "mathit" | "operatorname" | "boxed" => {
                        if let Some((arg, end)) = braced_group(&chars, j) {
                            out.push_str(&normalize_answer(&arg));
                            i = end;
                            prev_digit = false;
                            continue;
                        }
                    }
                    "pi" => out.push_str("pi"),
                    "tau" => out.push_str("tau"),
                    "cdot" | "times" => out.push('*'),
                    "div" => out.push('/'),
                    "left" | "right" | "quad" | "qquad" | "!" | "," | ";" | " " => {}
                    "%" => {}
                    "" => {
                        // Backslash followed by punctuation (e.g. \{).
                        if let Some(&n) = chars.get(j) {
                            out.push(n);
                            j += 1;
                        }
                    }
                    other => out.push_str(other),
                }
                i = j;
                prev_digit = false;
                continue;
            }
            '^' => {
                out.push('^');
                if chars.get(i + 1) == Some(&'{') {
                    if let Some((arg, end)) = braced_group(&chars, i + 1) {
                        out.push('(');
                        out.push_str(&normalize_answer(&arg));
                        out.push(')');
                        i = end;
                        prev_digit = false;
                        continue;
                    }
                }
                prev_digit = false;
            }
            '_' => {
                // x_1 -> x1 is safe for single-character subscripts.
                if chars.get(i + 1) == Some(&'{') {
                    if let Some((arg, end)) = braced_group(&chars, i + 1) {
                        out.push_str(&normalize_answer(&arg));
                        i = end;
                        prev_digit = false;
                        continue;
                    }
                }
            }
            '(' | ')' | '+' | '-' | '*' | '/' | '=' => {
                out.push(c);
                prev_digit = false;
            }
            '.' => {
                out.push(c);
                prev_digit = c.is_ascii_digit();
            }
            c if c.is_ascii_digit() || c.is_ascii_alphabetic() => {
                out.push(c);
                prev_digit = c.is_ascii_digit();
            }
            _ => {
                // Drop unknown symbols rather than letting them poison a parse.
            }
        }
        i += 1;
    }
    out
}

fn braced_group(chars: &[char], open: usize) -> Option<(String, usize)> {
    if open >= chars.len() || chars[open] != '{' {
        return None;
    }
    let mut depth = 0usize;
    for (i, &c) in chars.iter().enumerate().skip(open) {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((chars[open + 1..i].iter().collect(), i));
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Restricted arithmetic parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(char),
    LParen,
    RParen,
    Comma,
}

fn lex(input: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).map(|n| n.is_ascii_digit()).unwrap_or(false)) {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < chars.len() && chars[i] == '.' {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                let ds = j;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                if j > ds {
                    i = j;
                }
            }
            let text: String = chars[start..i].iter().collect();
            let value = text.parse::<f64>().map_err(|e| format!("bad number {text}: {e}"))?;
            out.push(Tok::Num(value));
            continue;
        }
        if c.is_ascii_alphabetic() {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '_')
            {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
            continue;
        }
        match c {
            '+' | '-' | '*' | '/' | '^' => out.push(Tok::Op(c)),
            '(' => out.push(Tok::LParen),
            ')' => out.push(Tok::RParen),
            ',' => out.push(Tok::Comma),
            other => return Err(format!("unexpected symbol '{other}'")),
        }
        i += 1;
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(f64),
    Var(String),
    Neg(Box<Node>),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Div(Box<Node>, Box<Node>),
    Pow(Box<Node>, Box<Node>),
    Func(String, Vec<Node>),
}

impl Node {
    fn vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Node::Var(name) => {
                out.insert(name.clone());
            }
            Node::Neg(a) => a.vars(out),
            Node::Func(_, args) => {
                for arg in args {
                    arg.vars(out);
                }
            }
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b) => {
                a.vars(out);
                b.vars(out);
            }
            Node::Num(_) => {}
        }
    }

    fn eval(&self, env: &std::collections::BTreeMap<String, f64>) -> f64 {
        match self {
            Node::Num(v) => *v,
            Node::Var(name) => *env.get(name).unwrap_or(&f64::NAN),
            Node::Neg(a) => -a.eval(env),
            Node::Add(a, b) => a.eval(env) + b.eval(env),
            Node::Sub(a, b) => a.eval(env) - b.eval(env),
            Node::Mul(a, b) => a.eval(env) * b.eval(env),
            Node::Div(a, b) => a.eval(env) / b.eval(env),
            Node::Pow(a, b) => a.eval(env).powf(b.eval(env)),
            Node::Func(name, args) => {
                let x = args.first().map(|a| a.eval(env)).unwrap_or(f64::NAN);
                match name.as_str() {
                    "sqrt" => x.sqrt(),
                    "abs" => x.abs(),
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    "ln" | "log" if args.len() == 1 => x.ln(),
                    "log" if args.len() == 2 => x.ln() / args[1].eval(env).ln(),
                    "log2" => x.log2(),
                    "exp" => x.exp(),
                    "floor" => x.floor(),
                    "ceil" => x.ceil(),
                    "round" => x.round(),
                    _ => f64::NAN,
                }
            }
        }
    }

    fn variable_set(&self) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        self.vars(&mut set);
        set
    }
}

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn new(input: &str) -> Result<Self, String> {
        Ok(Self { tokens: lex(input)?, pos: 0 })
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn starts_atom(&self) -> bool {
        matches!(self.peek(), Some(Tok::Num(_)) | Some(Tok::Ident(_)) | Some(Tok::LParen))
    }

    fn parse(mut self) -> Result<Node, String> {
        let node = self.expr()?;
        if self.pos != self.tokens.len() {
            return Err(format!("trailing tokens at {:?}", self.peek()));
        }
        Ok(node)
    }

    fn expr(&mut self) -> Result<Node, String> {
        let mut left = self.term()?;
        loop {
            match self.peek() {
                Some(Tok::Op('+')) => {
                    self.pos += 1;
                    let right = self.term()?;
                    left = Node::Add(Box::new(left), Box::new(right));
                }
                Some(Tok::Op('-')) => {
                    self.pos += 1;
                    let right = self.term()?;
                    left = Node::Sub(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Node, String> {
        let mut left = self.factor()?;
        loop {
            match self.peek() {
                Some(Tok::Op('*')) => {
                    self.pos += 1;
                    let right = self.factor()?;
                    left = Node::Mul(Box::new(left), Box::new(right));
                }
                Some(Tok::Op('/')) => {
                    self.pos += 1;
                    let right = self.factor()?;
                    left = Node::Div(Box::new(left), Box::new(right));
                }
                _ if self.starts_atom() => {
                    // Implicit multiplication: 2pi, 3(x+1), 2sqrt(2).
                    let right = self.factor()?;
                    left = Node::Mul(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn factor(&mut self) -> Result<Node, String> {
        match self.peek() {
            Some(Tok::Op('+')) => {
                self.pos += 1;
                self.factor()
            }
            Some(Tok::Op('-')) => {
                self.pos += 1;
                Ok(Node::Neg(Box::new(self.factor()?)))
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Node, String> {
        let base = self.atom()?;
        if matches!(self.peek(), Some(Tok::Op('^'))) {
            self.pos += 1;
            let exp = self.factor()?;
            return Ok(Node::Pow(Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Node, String> {
        match self.peek().cloned() {
            Some(Tok::Num(v)) => {
                self.pos += 1;
                Ok(Node::Num(v))
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.expr()?);
                            if matches!(self.peek(), Some(Tok::Comma)) {
                                self.pos += 1;
                                continue;
                            }
                            break;
                        }
                    }
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        return Err("missing closing parenthesis in function call".into());
                    }
                    self.pos += 1;
                    return Ok(Node::Func(name, args));
                }
                if name == "pi" || name == "tau" || name == "e" {
                    let value = match name.as_str() {
                        "pi" => std::f64::consts::PI,
                        "tau" => std::f64::consts::TAU,
                        _ => std::f64::consts::E,
                    };
                    return Ok(Node::Num(value));
                }
                Ok(Node::Var(name))
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let node = self.expr()?;
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    return Err("missing closing parenthesis".into());
                }
                self.pos += 1;
                Ok(node)
            }
            other => Err(format!("expected atom, got {other:?}")),
        }
    }
}

/// Numeric value of a string after normalization, if it is a constant expression.
pub fn numeric_value(text: &str) -> Option<f64> {
    let normalized = normalize_answer(text);
    let node = Parser::new(&normalized).ok()?.parse().ok()?;
    if !node.variable_set().is_empty() {
        return None;
    }
    let value = node.eval(&std::collections::BTreeMap::new());
    value.is_finite().then_some(value)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

/// True if two answers are the same after normalization and numeric evaluation.
pub fn math_equivalent(predicted: &str, expected: &str, tol: f64) -> bool {
    let a = normalize_answer(predicted);
    let b = normalize_answer(expected);
    if a == b {
        return true;
    }
    let pa = Parser::new(&a).and_then(Parser::parse);
    let pb = Parser::new(&b).and_then(Parser::parse);
    if let (Ok(pa), Ok(pb)) = (&pa, &pb) {
        if pa.variable_set() == pb.variable_set() {
            let vars: Vec<String> = pa.variable_set().into_iter().collect();
            let pool = [0.5_f64, 1.0, 2.0, -1.5, 3.0, 7.0];
            let mut checked = 0;
            for (idx, _) in pool.iter().enumerate() {
                let mut env = std::collections::BTreeMap::new();
                for (v_idx, name) in vars.iter().enumerate() {
                    env.insert(name.clone(), pool[(idx + v_idx) % pool.len()]);
                }
                let va = pa.eval(&env);
                let vb = pb.eval(&env);
                if !va.is_finite() || !vb.is_finite() {
                    continue; // singular point for this pair; try the next one
                }
                checked += 1;
                if !close(va, vb, tol) {
                    return false;
                }
            }
            if checked >= 2 {
                return true;
            }
        }
    }
    match (numeric_value(&a), numeric_value(&b)) {
        (Some(x), Some(y)) => close(x, y, tol),
        _ => false,
    }
}

/// Reward for a chain-of-thought response: 1.0 only when the extracted answer
/// is mathematically equivalent to the reference.
pub fn verify_math(response: &str, expected: &str, tol: f64) -> f32 {
    let answer = extract_answer(response);
    if math_equivalent(&answer, expected, tol) {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_boxed_and_final_line() {
        assert_eq!(extract_answer("reasoning... \\boxed{42}"), "42");
        assert_eq!(extract_answer("step\nThe answer is 3.14"), "3.14");
        assert_eq!(extract_answer("\\boxed{\\frac{1}{2}}"), "\\frac{1}{2}");
        assert_eq!(extract_answer("answer 42"), "42");
        assert_eq!(extract_answer("the result is 1/3"), "1/3");
    }

    #[test]
    fn normalizes_latex() {
        assert_eq!(normalize_answer("\\frac{1}{2}"), "(1)/(2)");
        assert_eq!(normalize_answer("2\\sqrt{2}"), "2sqrt(2)");
        assert_eq!(normalize_answer("1,000"), "1000");
        assert_eq!(normalize_answer("x^{2}+1"), "x^(2)+1");
    }

    #[test]
    fn numeric_equivalence_covers_common_forms() {
        let tol = 1e-5;
        assert!(math_equivalent("1/2", "0.5", tol));
        assert!(math_equivalent("\\frac{1}{2}", "0.5", tol));
        assert!(math_equivalent("2^{3}", "8", tol));
        assert!(math_equivalent("\\pi", "3.14159265", tol));
        assert!(math_equivalent("sqrt(2)", "1.41421356", tol));
        assert!(math_equivalent("2sqrt(2)", "2.8284271", tol));
        assert!(!math_equivalent("1/3", "0.5", tol));
    }

    #[test]
    fn symbolic_equivalence_at_sample_points() {
        assert!(math_equivalent("1+x", "x+1", 1e-6));
        assert!(math_equivalent("2*x", "x+x", 1e-6));
        assert!(!math_equivalent("x+1", "x+2", 1e-6));
    }

    #[test]
    fn verify_math_scores_chain_of_thought() {
        assert_eq!(verify_math("work... answer is \\boxed{0.5}", "1/2", 1e-5), 1.0);
        assert_eq!(verify_math("work... answer is 7", "1/2", 1e-5), 0.0);
    }

    #[test]
    fn rejects_unparseable_answers() {
        assert!(!math_equivalent("hello world", "42", 1e-5));
        assert_eq!(numeric_value("2+3"), Some(5.0));
        assert_eq!(numeric_value("x+1"), None);
    }
}
