// --------------------------------------------------------------------------------------------------------------------
// <copyright file="RandomInitializer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Helpers
{
    using System;

    /// <summary>
    /// Represents a pseudo-random generator for values used within the net.
    /// </summary>
    public class RandomInitializer
    {
        /// <summary>
        /// The inner generator.
        /// </summary>
        private readonly Random random = new Random();

        /// <summary>
        /// Returns a random floating point number (by default greater than or equal to -1.0, and less than 1.0).
        /// </summary>
        /// <param name="range">
        /// Size of the range.
        /// </param>
        /// <param name="offset">
        /// Offset of the beginning of the range relative to 0.0.
        /// </param>
        /// <returns>
        /// Random <see cref="float"/>.
        /// </returns>
        public float NextFloat(float range = 2f, float offset = -1f)
        {
            if (range < 0f)
            {
                throw new ArgumentException("Range size cannot be below zero.", nameof(range));
            }

            var result = (float)this.random.NextDouble();
            result *= range;
            result += offset;
            return result;
        }
    }
}
