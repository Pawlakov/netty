using System;
using System.Collections.Generic;
using System.Text;

namespace Netty.Net.Layers.Builders
{
    public class UnpoolingLayerBuilder : ILayerBuilder
    {
        private readonly int kernelHeight;

        private readonly int kernelWidth;

        public UnpoolingLayerBuilder(int kernelHeight, int kernelWidth)
        {
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
        }

        public ILayer Build(int inputDepth, int inputHeight, int inputWidth)
        {
            return new UnpoolingLayer(inputDepth, inputHeight, inputWidth, this.kernelHeight, this.kernelWidth);
        }
    }
}
