namespace Netty.Floater
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Drawing;
    using System.IO;

    using Microsoft.Extensions.Configuration;

    public class InputDataSerializer : IEnumerable<LazySerializedImage>
    {
        private readonly string[] files;

        private readonly int height;

        private readonly int width;
        
        public InputDataSerializer(IConfiguration configuration, int height, int width)
        {
            this.height = height;
            this.width = width;
            string path;
            try
            {
                path = configuration["DataDirectory"];
            }
            catch (Exception exception)
            {
                throw new DirectoryConfigurationException("Input data directory is not correctly configured.", exception);
            }

            if (string.IsNullOrEmpty(path))
            {
                throw new DirectoryConfigurationException("Input data directory path is missing.");
            }

            if (!Directory.Exists(path))
            {
                throw new DirectoryConfigurationException("Input data directory does not exist.");
            }

            try
            {
                this.files = Directory.GetFiles(path);
            }
            catch (UnauthorizedAccessException exception)
            {
                throw new DirectoryConfigurationException("You are not authorized to read input data directory.", exception);
            }
        }

        public IEnumerator<LazySerializedImage> GetEnumerator()
        {
            foreach (var file in this.files)
            {
                try
                {
                    using (var image = Image.FromFile(file))
                    {
                        if (image == null || image.Width != this.width || image.Height != this.height)
                        {
                            continue;
                        }
                    }
                }
                catch
                {
                    continue;
                }

                yield return new LazySerializedImage(file);
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}