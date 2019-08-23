namespace Netty.Net
{
    using Netty.Net.Helpers;

    public class ActivationLayer : ILayer
    {
        private readonly int height;

        private readonly int width;

        private readonly float[,] output;

        private readonly float[,] gradientCostOverInput;

        public ActivationLayer(int height, int width)
        {
            this.height = height;
            this.width = width;
            this.output = new float[height, width];
            this.gradientCostOverInput = new float[height, width];
        }

        public float[,] FeedForward(float[,] input)
        {
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.output[i, j] = ActivationHelper.Activation(input[i, j]);
                }
            }

            return this.output;
        }

        public float[,] BackPropagate(float[,] gradientCostOverOutput, float learningFactor = 1f)
        {
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.gradientCostOverInput[i, j] = gradientCostOverOutput[i, j] * ActivationHelper.ActivationGradient(this.output[i, j]);
                }
            }

            return this.gradientCostOverInput;
        }
    }
}