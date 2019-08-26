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

        /// <summary>
        /// Adds padding to the the matrix. Removes when negative amount is given.
        /// </summary>
        /// <param name="input">
        /// Input matrix.
        /// </param>
        /// <param name="output">
        /// Output matrix.
        /// </param>
        /// <param name="verticalPadding">
        /// The amount of padding along vertical edges.
        /// </param>
        /// <param name="horizontalPadding">
        /// The amount of padding along horizontal edges.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when one of matrices is null.
        /// </exception>
        /// <exception cref="MatrixException">
        /// Thrown when matrices dimensions do not support this operation.
        /// </exception>
        public static void Pad(float[,,] input, float[,,] output, int verticalPadding, int horizontalPadding)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            var depth = output.GetLength(0);
            var height = output.GetLength(1);
            var width = output.GetLength(2);
            if (input.GetLength(0) != depth
                || input.GetLength(1) != height - (verticalPadding * 2)
                || input.GetLength(2) != width - (horizontalPadding * 2))
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            var maxJ = verticalPadding > 0 ? height - verticalPadding : height;
            var maxK = horizontalPadding > 0 ? width - horizontalPadding : width;
            for (var i = 0; i < depth; ++i)
            {
                var j = verticalPadding > 0 ? verticalPadding : 0;
                for (; j < maxJ; ++j)
                {
                    var k = horizontalPadding > 0 ? horizontalPadding : 0;
                    for (; k < maxK; ++k)
                    {
                        output[i, j, k] = input[i, j - verticalPadding, k - horizontalPadding];
                    }
                }
            }
        }

        /// <summary>
        /// Flips the matrix horizontally and vertically.
        /// </summary>
        /// <param name="input">
        /// Input matrix.
        /// </param>
        /// <param name="output">
        /// Output matrix.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when one of matrices is null.
        /// </exception>
        /// <exception cref="MatrixException">
        /// Thrown when matrices dimensions do not support this operation.
        /// </exception>
        public static void Flip(float[,,,] input, float[,,,] output)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            var count = input.GetLength(0);
            var depth = input.GetLength(1);
            var height = input.GetLength(2);
            var width = input.GetLength(3);
            if (output.GetLength(1) != count || output.GetLength(0) != depth || output.GetLength(2) != height || output.GetLength(3) != width)
            {
                throw new MatrixException("Matrices dimensions do not support this operation.");
            }

            for (var i = 0; i < count; ++i)
            {
                for (var j = 0; j < depth; ++j)
                {
                    for (var k = 0; k < height; ++k)
                    {
                        for (var l = 0; l < width; ++l)
                        {
                            output[j, i, height - k - 1, width - l - 1] = input[i, j, k, l];
                        }
                    }
                }
            }
        }
    }
}
