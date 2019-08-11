namespace Netty
{
    using Netty.Net;
    using Netty.Net.Helpers;
    using System;

    public static class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Starting.");
            var random = new RandomInitializer();
            var input = new float[4, 4];
            for (var i = 0; i < 4; ++i)
            {
                for (var j = 0; j < 4; ++j)
                {
                    input[i, j] = i + j;
                }
            }

            var layer = new ConvolutionLayer(4, 4);
            var output = layer.Calculate(input);
            for (var i = 0; i < 4; ++i)
            {
                Console.Write("[");
                for (var j = 0; j < 4; ++j)
                {
                    Console.Write("{0:0.00} ", input[i, j]);
                }

                Console.WriteLine("]");
            }
            
            Console.WriteLine();
            for (var i = 0; i < 4; ++i)
            {
                Console.Write("[");
                for (var j = 0; j < 4; ++j)
                {
                    Console.Write("{0:0.00} ", output[i, j]);
                }
                
                Console.WriteLine("]");
            }
        }
    }
}
