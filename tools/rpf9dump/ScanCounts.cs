using System.Text.RegularExpressions;

// Scan-level counts for the manifest. Candidates that RecoverAtOffsets rejects never reach Program.cs, so
// Candidates - Recovered is the only place those rejects show up.
internal static class ScanCounts
{
    // Written into the index cache's coverage text on a full scan.
    public static string Coverage(int candidates, int recovered) =>
        FormattableString.Invariant($"rpf9dump full scan, {candidates} candidates, {recovered} recovered");

    // Reads the candidate count back from a cached index. Older caches wrote culture-formatted numbers
    // ("183,801"), so digit-group separators are dropped. Null when the text is not ours or not readable.
    public static int? Candidates(string? coverage)
    {
        if (coverage is null) return null;
        var m = Regex.Match(coverage, @"^rpf9dump full scan, ([0-9][0-9.,' \u00A0\u202F]*) candidates,");
        if (!m.Success) return null;
        string digits = new(m.Groups[1].Value.Where(char.IsAsciiDigit).ToArray());
        return int.TryParse(digits, out int n) ? n : null;
    }
}
