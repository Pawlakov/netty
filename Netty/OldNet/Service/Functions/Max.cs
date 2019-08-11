namespace ClickbaitGenerator.NeuralNet.Service.Functions
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class Max : IPoolingFunction
    {
        public float Calculate(IList<float> input, out int index)
        {
            var max = float.MinValue;
            float currentActivation;
            var emptyConnectionsOnWay = 0; // Counts how many connections that are half-empty are there (have no input neuron)
            index = -1;

            for (var i = 0; i < input.Count; i++)
            {
                currentActivation = input[i];
                if (currentActivation > max)
                {
                    max = currentActivation;
                    index = i;
                }
            }

            for (var i = 0; i < index; i++)
            {
                currentActivation = input[i];
                if (currentActivation < 0.0f)
                {
                    emptyConnectionsOnWay++;
                }
            }

            index -= emptyConnectionsOnWay;
            return max;
        }
    }
}
