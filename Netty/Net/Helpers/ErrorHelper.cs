// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ErrorHelper.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Helpers
{
    using System;

    using Netty.Net.Exceptions;

    /// <summary>
    /// Container for methods for error calculation.
    /// </summary>
    public static class ErrorHelper
    {
        /// <summary>
        /// Calculates the mean squared error.
        /// </summary>
        /// <param name="template">
        /// Expected output.
        /// </param>
        /// <param name="subject">
        /// Actual output.
        /// </param>
        /// <returns>
        /// The <see cref="float"/> representing the error value.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when one of the parameters is null.
        /// </exception>
        /// <exception cref="MatrixException">
        /// Thrown when the dimensions of the matrices do not align.
        /// </exception>
        public static float CalculateError(float[,] template, float[,] subject)
        {
            if (template == null)
            {
                throw new ArgumentNullException(nameof(template));
            }

            if (subject == null)
            {
                throw new ArgumentNullException(nameof(subject));
            }

            var firstDimension = template.GetLength(0);
            var secondDimension = template.GetLength(1);
            if (firstDimension != subject.GetLength(0) || secondDimension != subject.GetLength(1))
            {
                throw new MatrixException("The dimensions of these matrices do not align.");
            }

            var sum = 0f;
            var n = firstDimension * secondDimension;
            for (var i = 0; i < firstDimension; ++i)
            {
                for (var j = 0; j < secondDimension; ++j)
                {
                    var difference = template[i, j] - subject[i, j];
                    sum += difference * difference;
                }
            }

            return sum / n;
        }

        /// <summary>
        /// Calculates error gradient of the mean squared error function with regard to output.
        /// </summary>
        /// <param name="template">
        /// Expected output.
        /// </param>
        /// <param name="subject">
        /// Actual output.
        /// </param>
        /// <param name="gradient">
        /// Array for the resulting gradient.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when one of the parameters is null.
        /// </exception>
        /// <exception cref="MatrixException">
        /// Thrown when the dimensions of the matrices do not align.
        /// </exception>
        public static void CalculateErrorGradient(float[,] template, float[,] subject, float[,] gradient)
        {
            if (template == null)
            {
                throw new ArgumentNullException(nameof(template));
            }

            if (subject == null)
            {
                throw new ArgumentNullException(nameof(subject));
            }

            if (gradient == null)
            {
                throw new ArgumentNullException(nameof(gradient));
            }

            var firstDimension = template.GetLength(0);
            var secondDimension = template.GetLength(1);
            if (firstDimension != subject.GetLength(0) || secondDimension != subject.GetLength(1) || firstDimension != gradient.GetLength(0) || secondDimension != gradient.GetLength(1))
            {
                throw new MatrixException("The dimensions of these matrices do not align.");
            }

            var n = firstDimension * secondDimension;
            for (var i = 0; i < firstDimension; ++i)
            {
                for (var j = 0; j < secondDimension; ++j)
                {
                    gradient[i, j] = (-2f / n) * (template[i, j] - subject[i, j]);
                }
            }
        }
    }
}