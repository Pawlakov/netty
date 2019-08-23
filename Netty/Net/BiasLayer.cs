namespace Netty.Net
{
    public class BiasLayer : ILayer
    {
        private readonly int height;

        private readonly int width;

        private float bias;

        private readonly float[,] output;

        private readonly float[,] gradientCostOverInput;

        public BiasLayer(int height, int width)
        {
            this.height = height;
            this.width = width;
            this.bias = 0f;
            this.output = new float[height, width];
            this.gradientCostOverInput = new float[height, width];
        }

        public float[,] FeedForward(float[,] input)
        {
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.output[i, j] = input[i, j] + this.bias;
                }
            }

            return this.output;
        }

        public float[,] BackPropagate(float[,] gradientCostOverOutput, float learningFactor = 1f)
        {
            // Calculate bias and input gradient.
            var gradientCostOverBias = 0f;
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.gradientCostOverInput[i, j] = gradientCostOverOutput[i, j];
                    gradientCostOverBias += gradientCostOverOutput[i, j];
                }
            }

            // Apply bias gradient.
            this.bias -= learningFactor * gradientCostOverBias;

            // Return input gradient.
            return this.gradientCostOverInput;
        }
    }
}