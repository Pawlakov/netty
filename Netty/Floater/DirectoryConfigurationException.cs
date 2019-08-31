namespace Netty.Floater
{
    using System;

    public class DirectoryConfigurationException : Exception
    {
        public DirectoryConfigurationException()
            : this("The directory is incorrectly configured.")
        {
        }

        public DirectoryConfigurationException(string message)
            : base($"{message} Check the App.config file.")
        {
        }


        public DirectoryConfigurationException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}