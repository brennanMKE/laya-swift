// Ported from laya-go internal/tokenizer/pretokenize.go.
//
// Everything here works on UTF-8 bytes and Unicode scalars, never on
// Character: a Character is a grapheme cluster ("\r\n" is one), while the
// GPT-2 pattern is defined on code points.

/// GPT-2's bytes_to_unicode table: printable Latin-1 bytes map to themselves
/// and the rest to U+0100 onward, so every byte has a visible character
/// (space is "Ġ", newline "Ċ").
let byteToScalar: [Unicode.Scalar] = {
    var table = [Unicode.Scalar](repeating: " ", count: 256)
    var n: UInt32 = 0
    for b in 0..<256 {
        if (0x21...0x7E).contains(b) || (0xA1...0xAC).contains(b) || (0xAE...0xFF).contains(b) {
            table[b] = Unicode.Scalar(UInt8(b))
        } else {
            table[b] = Unicode.Scalar(256 + n)!
            n += 1
        }
    }
    return table
}()

extension Unicode.Scalar {
    /// \s in the Oniguruma pattern the Rust tokenizers use: the Unicode
    /// White_Space property, which is what Go's unicode.IsSpace reports.
    var isPatternSpace: Bool {
        if isASCII { return value == 0x20 || (0x09...0x0D).contains(value) }
        return properties.isWhitespace
    }

    /// \p{L}: general category L (Lu, Ll, Lt, Lm, Lo).
    var isPatternLetter: Bool {
        if isASCII { return (0x41...0x5A).contains(value) || (0x61...0x7A).contains(value) }
        switch properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    /// \p{N}: general category N (Nd, Nl, No).
    var isPatternNumber: Bool {
        if isASCII { return (0x30...0x39).contains(value) }
        switch properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }

    /// [^\s\p{L}\p{N}].
    var isPatternOther: Bool { !isPatternSpace && !isPatternLetter && !isPatternNumber }
}

/// Decodes the scalar starting at `bytes[i]`, returning it and its length.
/// The bytes come from a Swift String, so they are valid UTF-8.
@inline(__always)
func decodeScalar(_ bytes: [UInt8], at i: Int) -> (Unicode.Scalar, Int) {
    let b0 = UInt32(bytes[i])
    let value: UInt32
    let size: Int
    switch b0 {
    case ..<0x80:
        return (Unicode.Scalar(UInt8(b0)), 1)
    case ..<0xE0:
        value = (b0 & 0x1F) << 6 | UInt32(bytes[i + 1] & 0x3F)
        size = 2
    case ..<0xF0:
        value = (b0 & 0x0F) << 12 | UInt32(bytes[i + 1] & 0x3F) << 6 | UInt32(bytes[i + 2] & 0x3F)
        size = 3
    default:
        value = (b0 & 0x07) << 18 | UInt32(bytes[i + 1] & 0x3F) << 12
            | UInt32(bytes[i + 2] & 0x3F) << 6 | UInt32(bytes[i + 3] & 0x3F)
        size = 4
    }
    return (Unicode.Scalar(value) ?? "\u{FFFD}", size)
}

/// Decodes the scalar that ends at `bytes[end - 1]`, returning it and its length.
@inline(__always)
func decodeLastScalar(_ bytes: [UInt8], before end: Int) -> (Unicode.Scalar, Int) {
    var start = end - 1
    while start > 0 && bytes[start] & 0xC0 == 0x80 { start -= 1 }
    return (decodeScalar(bytes, at: start).0, end - start)
}

/// Splits `bytes[range]` the way the GPT-2 pattern does,
///
///     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
///
/// calling `emit` with each piece in order. Alternatives are tried in order
/// at each position and each is greedy, as in a backtracking regex engine.
func preTokenize(_ bytes: [UInt8], _ range: Range<Int>, emit: (Range<Int>) -> Void) {
    var i = range.lowerBound
    while i < range.upperBound {
        let n = matchAt(bytes, i, range.upperBound)
        emit(i..<(i + n))
        i += n
    }
}

/// The length in bytes of the pattern's match at `bytes[i..<end]`. Every
/// scalar is matched by some alternative, so the result is always > 0.
func matchAt(_ bytes: [UInt8], _ i: Int, _ end: Int) -> Int {
    let rest = end - i
    // 's|'t|'re|'ve|'m|'ll|'d
    if bytes[i] == UInt8(ascii: "'") && rest > 1 {
        switch bytes[i + 1] {
        case UInt8(ascii: "s"), UInt8(ascii: "t"), UInt8(ascii: "m"), UInt8(ascii: "d"):
            return 2
        case UInt8(ascii: "r"), UInt8(ascii: "v"):
            if rest > 2 && bytes[i + 2] == UInt8(ascii: "e") { return 3 }
        case UInt8(ascii: "l"):
            if rest > 2 && bytes[i + 2] == UInt8(ascii: "l") { return 3 }
        default:
            break
        }
    }
    // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`
    for kind in [CharacterKind.letter, .number, .other] {
        let n = optionalSpaceRun(bytes, i, end, kind)
        if n > 0 { return n }
    }
    // \s+(?!\S) then \s+. A whitespace run followed by a non-space scalar
    // gives up its last scalar to the next token (so " word" keeps its
    // space), unless the run is a single scalar, which \s+ then takes.
    var j = i, last = i, count = 0
    while j < end {
        let (scalar, size) = decodeScalar(bytes, at: j)
        if !scalar.isPatternSpace { break }
        last = j
        j += size
        count += 1
    }
    if count == 0 {
        // Unreachable: isPatternOther covers every other scalar.
        return decodeScalar(bytes, at: i).1
    }
    if j == end || count == 1 { return j - i }
    return last - i
}

enum CharacterKind {
    case letter, number, other

    @inline(__always)
    func matches(_ scalar: Unicode.Scalar) -> Bool {
        switch self {
        case .letter: scalar.isPatternLetter
        case .number: scalar.isPatternNumber
        case .other: scalar.isPatternOther
        }
    }
}

/// Matches ` ?kind+` at `bytes[i]` and returns its length, or 0.
private func optionalSpaceRun(_ bytes: [UInt8], _ i: Int, _ end: Int, _ kind: CharacterKind) -> Int {
    var j = bytes[i] == UInt8(ascii: " ") ? i + 1 : i
    let start = j
    while j < end {
        let (scalar, size) = decodeScalar(bytes, at: j)
        if !kind.matches(scalar) { break }
        j += size
    }
    return j == start ? 0 : j - i
}
