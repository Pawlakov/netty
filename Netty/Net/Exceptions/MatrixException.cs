// --------------------------------------------------------------------------------------------------------------------
// <copyright file="MatrixException.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Exceptions
{
    using System;

    /// <summary>
    /// Represents error occurring during a matrix operation.
    /// </summary>
    public class MatrixException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MatrixException"/> class.
        /// </summary>
        public MatrixException()
            : base("Matrix operation failed.")
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatrixException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">
        /// The message.
        /// </param>
        public MatrixException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatrixException"/> class with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">
        /// The message.
        /// </param>
        /// <param name="innerException">
        /// The inner exception.
        /// </param>
        public MatrixException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}