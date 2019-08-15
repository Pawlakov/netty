// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ActivationHelper.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Helpers
{
    using System;

    /// <summary>
    /// Container for methods regarding the activation function.
    /// </summary>
    public static class ActivationHelper
    {
        /// <summary>
        /// The sigmoid activation function.
        /// </summary>
        /// <param name="value">
        /// Input value.
        /// </param>
        /// <returns>
        /// The <see cref="float"/> activation value.
        /// </returns>
        public static float Activation(float value)
        {
            var expResult = Math.Exp(-value);
            var result = 1 / (float)(1 + expResult);
            return result;
        }

        /// <summary>
        /// The gradient of sigmoid activation function.
        /// </summary>
        /// <param name="value">
        /// Input value.
        /// </param>
        /// <returns>
        /// The <see cref="float"/> gradient value.
        /// </returns>
        public static float ActivationGradient(float value)
        {
            var sigma = Activation(value);
            return sigma * (1 - sigma);
        }
    }
}
