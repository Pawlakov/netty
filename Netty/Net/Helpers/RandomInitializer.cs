namespace Netty.Net.Helpers
{
    using System;

    public class RandomInitializer
    {
        private readonly Random random = new Random();

        public float NextFloat(int range = 2, int offset = -1)
        {
            float result = (float)random.NextDouble();
            result *= range;
            result += offset;
            return result;
        }
    }
}
