namespace Netty
{
    using Netty.Net;
    using Netty.Net.Helpers;
    using System;

    public static class Program
    {
        public static void Main(string[] args)
        {
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
            Console.WriteLine(input);
            Console.WriteLine(output);
        }
    }
}
