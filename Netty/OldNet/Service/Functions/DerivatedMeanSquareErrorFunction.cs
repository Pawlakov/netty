namespace ClickbaitGenerator.NeuralNet.Service.Functions
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class DerivatedMeanSquareErrorFunction : IErrorFunction
    {
        /// <summary>
        /// Multiplies the resulting single errors.
        /// </summary>
        private float _factor;

        public string Name { get; } = "Derivated Mean Square Error";
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="errorScalingFactor">Each single error will be multiplied by this value.</param>
        public DerivatedMeanSquareErrorFunction(float errorScalingFactor = 1.0f)
        {
            this._factor = errorScalingFactor;
        }

        public List<float> CalculateErrors(List<float> expectedValues, List<float> outcomeValues)
        {
            var valuesCount = expectedValues.Count;
            var errorsList = new List<float>(valuesCount); // All three lists in here have same size.

            for (var i = 0; i < valuesCount; i++)
            {
                var difference = expectedValues[i] - outcomeValues[i];
                errorsList.Add(/*difference **/ difference);
            }

            return errorsList;
        }
    }
}
