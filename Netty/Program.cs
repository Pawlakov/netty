namespace Netty
{
    using System;

    using Netty.Net;
    using Netty.Net.Helpers;

    public static class Program
    {
        public static void Main(string[] args)
        {
            var input = new float[4, 4];
            for (var i = 0; i < 4; ++i)
            {
                for (var j = 0; j < 4; ++j)
                {
                    input[i, j] = 0.1f * (i + j);
                }
            }

            var gradient = new float[4, 4];
            var layer = new ConvolutionLayer(4);
            while (true)
            {
                var output = layer.FeedForward(input);
                ErrorHelper.CalculateErrorGradient(input, output, gradient);
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

                Console.WriteLine("Error: {0:0.0000}", ErrorHelper.CalculateError(input, output));
                Console.ReadKey();
                Console.Clear();
                ErrorHelper.CalculateErrorGradient(input, output, gradient);
                layer.BackPropagate(gradient);
            }
        }
    }
}
