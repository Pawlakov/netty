namespace ClickbaitGenerator.NeuralNet.Helpers
{
    using System;

    public static class CustomRandom
    {
        static Random _random = new Random();
        /// <summary>
        /// Returns a random float number from given range of given offset.
        /// </summary>
        /// <param name="range"></param>
        /// <param name="offset"></param>
        /// <returns></returns>
        public static float NextFloat(int range = 2, int offset = 1)
        {
            float result = (float)_random.NextDouble();
            result *= range;
            result -= offset;

            return result;
        }
    }
}
