namespace Netty.Net.Helpers
{
    using System;

    public static class MatrixHelper
    {
        public static void Multiply(float[,] a, float[,] b, float[,] output, Func<float, float> postMap)
        {
            for (int i = 0; i < output.GetLength(0); ++i)
            {
                for (int j = 0; j < output.GetLength(1); ++j)
                {
                    var sum = 0f;
                    for (int k = 0; k < a.GetLength(1); ++k)
                    {
                        sum += a[i, k] * b[k, j];
                    }

                    output[i, j] = postMap(sum);
                }
            }
        }
    }
}
