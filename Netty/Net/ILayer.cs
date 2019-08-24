// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ILayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net
{
    /// <summary>
    /// Defines the interface of a sequential model layer.
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// Feeds forward the values through the layer.
        /// </summary>
        /// <param name="input">
        /// The input matrix.
        /// </param>
        /// <returns>
        /// The output matrix.
        /// </returns>
        float[,] FeedForward(float[,] input);

        /// <summary>
        /// Back-propagates the gradient of cost.
        /// </summary>
        /// <param name="gradientCostOverOutput">
        /// The gradient of cost over this layer's output.
        /// </param>
        /// <param name="learningFactor">
        /// The learning factor.
        /// </param>
        /// <returns>
        /// The gradient of cost over this layer's input.
        /// </returns>
        float[,] BackPropagate(float[,] gradientCostOverOutput, float learningFactor = 1f);
    }
}