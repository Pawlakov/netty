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
            Pad(input, output, padding, padding);
        }

        public static void Pad(float[,] input, float[,] output, int verticalPadding, int horizontalPadding)
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
            if (input.GetLength(0) != firstDimension - (verticalPadding * 2)
                || input.GetLength(1) != secondDimension - (horizontalPadding * 2))
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            var maxI = verticalPadding > 0 ? firstDimension - verticalPadding : firstDimension;
            var maxJ = horizontalPadding > 0 ? secondDimension - horizontalPadding : secondDimension;
            var i = verticalPadding > 0 ? verticalPadding : 0;
            for (; i < maxI; ++i)
            {
                var j = horizontalPadding > 0 ? horizontalPadding : 0;
                for (; j < maxJ; ++j)
                {
                    output[i, j] = input[i - verticalPadding, j - horizontalPadding];
                }
            }
        }

        public static void Flip(float[,] input, float[,] output)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            var firstDimension = input.GetLength(0);
            var secondDimension = input.GetLength(1);
            if (output.GetLength(0) != firstDimension || output.GetLength(1) != secondDimension)
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < firstDimension; ++i)
            {
                for (var j = 0; j < secondDimension; ++j)
                {
                    output[firstDimension - i - 1, secondDimension - j - 1] = input[i, j];
                }
            }
        }
    }
}
