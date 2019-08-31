namespace Netty.Floater
{
    using System;
    using System.Collections.Generic;

    public static class Toolkit
    {
        public static IEnumerable<float> Interlace24To32(this IEnumerable<float> first)
        {
            using (var enumerator1 = first.GetEnumerator())
            {
                while (enumerator1.MoveNext())
                {
                    yield return enumerator1.Current;
                    enumerator1.MoveNext();
                    yield return enumerator1.Current;
                    enumerator1.MoveNext();
                    yield return enumerator1.Current;
                    yield return 1.0f;
                }
            }
        }

        public static float ToFloat(byte value)
        {
            const float Factor = 1 / 255.0f;
            return value * Factor;
        }

        public static byte ToByte(float value)
        {
            return (byte)Math.Round(value * 255.0f, 0);
        }
    }
}