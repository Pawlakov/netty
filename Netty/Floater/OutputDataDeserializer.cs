namespace Netty.Floater
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;

    using Microsoft.Extensions.Configuration;

    public class OutputDataDeserializer
    {
        private readonly int height;

        private readonly int width;

        private readonly string outputDirectory;
        
        public OutputDataDeserializer(IConfiguration configuration, int height, int width)
        {
            this.height = height;
            this.width = width;
            try
            {
                this.outputDirectory = configuration["OutputDirectory"];
            }
            catch (Exception exception)
            {
                throw new DirectoryConfigurationException("Output directory is not correctly configured.", exception);
            }

            if (string.IsNullOrEmpty(this.outputDirectory))
            {
                throw new DirectoryConfigurationException("Output directory path is missing.");
            }

            if (!Directory.Exists(this.outputDirectory))
            {
                Directory.CreateDirectory(this.outputDirectory);
            }
        }

        public Bitmap ToBitmap(float[,,] floated)
        {
            var bitmap = new Bitmap(this.width, this.height, PixelFormat.Format32bppArgb);

            var rectangle = new Rectangle(0, 0, this.width, this.height);
            var bitmapData = bitmap.LockBits(rectangle, ImageLockMode.WriteOnly, bitmap.PixelFormat);

            var pointer = bitmapData.Scan0;

            var bytes = bitmapData.Stride * bitmap.Height;
            var data = new byte[bytes];
            for (var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < bitmap.Height; ++j)
                {
                    for (var k = 0; k < bitmap.Width; ++k)
                    {
                        data[(j * bitmapData.Stride) + (k * 4) + i] = Toolkit.ToByte(floated[i, j, k]);
                    }
                }
            }

            Marshal.Copy(data, 0, pointer, bytes);

            bitmap.UnlockBits(bitmapData);

            return bitmap;
        }
        
        public void Save(float[,,] floated)
        {
            var bitmap = this.ToBitmap(floated);

            var timeStamp = DateTime.UtcNow.ToString("dd-MM-yyyy HH-mm-ss-fff");
            var path = $"{this.outputDirectory}\\{timeStamp}.png";
            bitmap.Save(path, ImageFormat.Jpeg);
        }
    }
}