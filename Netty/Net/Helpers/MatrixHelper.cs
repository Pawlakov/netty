// --------------------------------------------------------------------------------------------------------------------
// <copyright file="MatrixHelper.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Helpers
{
    using System;

    using Netty.Net.Exceptions;

    /// <summary>
    /// Container for matrix manipulation helpers.
    /// </summary>
    public static class MatrixHelper
    {
        /// <summary>
        /// Matrix multiplication.
        /// </summary>
        /// <param name="a">
        /// First input matrix.
        /// </param>
        /// <param name="b">
        /// Second input matrix.
        /// </param>
        /// <param name="output">
        /// Output matrix.
        /// </param>
        /// <param name="postMap">
        /// Function transforming the result value before inserting it into the output matrix. Omit for plain assignment.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when one of the matrices is null.
        /// </exception>
        /// <exception cref="MatrixException">
        /// Thrown when matrices dimensions do not support multiplication.
        /// </exception>
        public static void Multiply(float[,] a, float[,] b, float[,] output, Func<float, float> postMap = null)
        {
            if (postMap == null)
            {
                postMap = x => x;
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            var firstDimension = output.GetLength(0);
            var secondDimension = output.GetLength(1);
            var commonDimension = a.GetLength(1);
            if (commonDimension != b.GetLength(0) || a.GetLength(0) != firstDimension || b.GetLength(1) != secondDimension)
            {
                throw new MatrixException("Matrices dimensions do not support multiplication.");
            }

            try
            {
                for (var i = 0; i < firstDimension; ++i)
                {
                    for (var j = 0; j < secondDimension; ++j)
                    {
                        var sum = 0f;
                        for (var k = 0; k < commonDimension; ++k)
                        {
                            sum += a[i, k] * b[k, j];
                        }

                        output[i, j] = postMap(sum);
                    }
                }
            }
            catch (Exception exception)
            {
                throw new MatrixException("Multiplication failed (probably due to wacky postMap delegate). See inner exception.", exception);
            }
        }

        public static void Pad(float[,] input, float[,] output, int padding)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            if (padding < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(padding), padding, "Padding cannot be below 0.");
            }

            var firstDimension = input.GetLength(0);
            var secondDimension = input.GetLength(1);
            if (output.GetLength(0) != firstDimension + (padding * 2) || output.GetLength(1) != secondDimension + (padding * 2))
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < firstDimension; ++i)
            {
                for (var j = 0; j < secondDimension; ++j)
                {
                    output[i + padding, j + padding] = input[i, j];
                }
            }
        }

        public static void UndoPadding(float[,] input, float[,] output, int padding)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            if (padding < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(padding), padding, "Padding cannot be below 0.");
            }

            var firstDimension = output.GetLength(0);
            var secondDimension = output.GetLength(1);
            if (input.GetLength(0) != firstDimension + (padding * 2) || input.GetLength(1) != secondDimension + (padding * 2))
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < firstDimension; ++i)
            {
                for (var j = 0; j < secondDimension; ++j)
                {
                    output[i, j] = input[i + padding, j + padding];
                }
            }
        }

        public static void UnfoldConvolutionInput(float[,] input, float[,] output, int filterDiameter)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            if (filterDiameter < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(filterDiameter), filterDiameter, "Filter diameter cannot be below 1.");
            }

            var firstDimension = input.GetLength(0);
            var secondDimension = input.GetLength(1);
            if (filterDiameter > firstDimension || filterDiameter > secondDimension)
            {
                throw new ArgumentOutOfRangeException(nameof(filterDiameter), filterDiameter, "Filter cannot be bigger than input.");
            }

            if (filterDiameter * filterDiameter > output.GetLength(1) || (firstDimension - filterDiameter + 1) * (secondDimension - filterDiameter + 1) > output.GetLength(0))
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < (firstDimension - filterDiameter + 1) * (secondDimension - filterDiameter + 1); ++i)
            {
                for (var j = 0; j < filterDiameter * filterDiameter; ++j)
                {
                    var x = (i / (secondDimension - filterDiameter + 1)) + (j / filterDiameter);
                    var y = (i % (secondDimension - filterDiameter + 1)) + (j % filterDiameter);
                    output[i, j] = input[x, y];
                }
            }
        }

        public static void UnfoldConvolutionFilter(float[,] input, float[,] output)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            var filterDiameter = input.GetLength(0);
            if (input.GetLength(1) != filterDiameter || filterDiameter * filterDiameter > output.GetLength(0))
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < filterDiameter; ++i)
            {
                for (var j = 0; j < filterDiameter; ++j)
                {
                    output[(i * filterDiameter) + j, 0] = input[i, j];
                }
            }
        }

        public static void FoldConvolutionOutput(float[,] input, float[,] output)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            var firstDimension = output.GetLength(0);
            var secondDimension = output.GetLength(1);
            if (input.GetLength(0) != firstDimension * secondDimension)
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < firstDimension; ++i)
            {
                for (var j = 0; j < secondDimension; ++j)
                {
                    output[i, j] = input[(i * secondDimension) + j, 0];
                }
            }
        }
    }
}
