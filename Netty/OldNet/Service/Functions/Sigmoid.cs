namespace ClickbaitGenerator.NeuralNet.Service.Functions
{
    using System;

    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class Sigmoid : IActivationFunction
    {
        private readonly float _beta;

        public Sigmoid(float beta = 1)
        {
            this._beta = beta;
        }

        public float Calculate(float input)
        {
            var expResult = Math.Exp(-this._beta * input);
            var result = 1 / (float)(1 + expResult);
            return result;
        }
    }
}
