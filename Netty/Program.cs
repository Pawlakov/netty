namespace Netty
{
    using Netty.Net;
    using Netty.Net.Helpers;
    using System;

    public static class Program
    {
        public static void Main(string[] args)
        {
            while (true)
            {
                var input = new float[4, 4];
                for (var i = 0; i < 4; ++i)
                {
                    for (var j = 0; j < 4; ++j)
                    {
                        input[i, j] = 0.1f * (i + j);
                    }
                }

                var layer = new ConvolutionLayer(4, 4);
                var output = layer.FeedForward(input);
                for (var i = 0; i < 4; ++i)
                {
                    Console.Write("[");
                    for (var j = 0; j < 4; ++j)
                    {
                        Console.Write("{0:0.0000} ", input[i, j]);
                    }

                    Console.WriteLine("]");
                }

                Console.WriteLine();
                for (var i = 0; i < 4; ++i)
                {
                    Console.Write("[");
                    for (var j = 0; j < 4; ++j)
                    {
                        Console.Write("{0:0.0000} ", output[i, j]);
                    }

                    Console.WriteLine("]");
                }

                Console.WriteLine("Error: {0:0.0000}", layer.CalculateError(input));
                Console.ReadKey();
                Console.Clear();
            }
        }
    }
}
