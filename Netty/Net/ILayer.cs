namespace Netty.Net
{
    public interface ILayer
    {
        float[,] FeedForward(float[,] input);

        float[,] BackPropagate(float[,] gradientCostOverOutput, float learningFactor = 1f);
    }
}