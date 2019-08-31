using System;

namespace Netty.Net.Layers
{
    public class DenseLayer : ILayer
    {
        private readonly int depth;

        private readonly int height;

        private readonly int width;

        private readonly int outputDepth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        public int OutputDepth => this.outputDepth;

        public int OutputHeight => this.outputHeight;

        public int OutputWidth => this.outputWidth;

        public DenseLayer(int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth)
        {
            this.depth = inputDepth;
            this.height = inputHeight;
            this.width = inputWidth;
            this.outputDepth = outputDepth;
            this.outputHeight = outputHeight;
            this.outputWidth = outputWidth;
        }

        public float[,,] FeedForward(float[,,] input)
        {
            throw new NotImplementedException();
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 1)
        {
            throw new NotImplementedException();
        }

        public void UpdateParameters()
        {
            throw new NotImplementedException();
        }
    }
}
