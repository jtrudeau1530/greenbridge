using System.Buffers.Binary;
using RpfPreview.Core;

Expect([], 4, "unrecognized-header");
Expect([], RageModelDecoder.MaxGraphicsBytes + 1, "graphics-size-limit");

// A synthetically truncated resource prefix containing a system-page pointer.
var prefix = new byte[16];
BinaryPrimitives.WriteUInt64LittleEndian(prefix.AsSpan(8), 0x50000080);
Expect(prefix, 16, "truncated-system");

// Recognizable drawable header with a deliberately malformed page table.
var header = new byte[0x100];
foreach (int at in new[] { 8, 16, 80, 168 })
    BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(at), 0x500000c0);
BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(44), 2);
foreach (int at in new[] { 64, 68, 72 }) BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(at), 1);
Expect(header, 16, "invalid-or-unsupported-layout");
Console.WriteLine("Model decode diagnostic smoke tests passed.");

static void Expect(ReadOnlySpan<byte> system, int graphicsBytes, string expected)
{
    string actual = ModelDecodeFailure.Explain(system, graphicsBytes).Code;
    if (actual != expected) throw new Exception($"Expected {expected}, got {actual}");
}
