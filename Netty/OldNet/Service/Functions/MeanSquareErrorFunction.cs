using System;

namespace ClickbaitGenerator.NeuralNet.Service.Functions
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class MeanSquareErrorFunction : IErrorFunction
    {
        /// <summary>
        /// Multiplier that multiplies resulting error. Usually 1/2.
        /// </summary>
        private float _multiplier;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="multiplier">Factor which the resulting single errors will be multiplied by.</param>
        public MeanSquareErrorFunction(float multiplier = 0.5f)
        {
            this._multiplier = multiplier;
        }

        public string Name { get; } = "Mean Square Error";
        
        public List<float> CalculateErrors(List<float> expectedValues, List<float> outcomeValues)
        {
            var errorsCount = expectedValues.Count;
            var errors = new List<float>(errorsCount);

            for (var i = 0; i < errorsCount; i++)
            {
                var difference = expectedValues[i] - outcomeValues[i];
                errors.Add(Math.Sign(difference) * (_multiplier * difference * difference));
            }

            return errors;
        }
    }
}
