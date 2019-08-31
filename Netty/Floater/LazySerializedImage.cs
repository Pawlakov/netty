namespace Netty.Floater
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;

    public class LazySerializedImage : IDisposable
    {
        private readonly string path;

        private bool open;

        private Action disposeAction;

        internal LazySerializedImage(string path)
        {
            this.path = path;
            this.open = false;
        }

        public float[,,] FloatedContent { get; private set; }

        public void Open()
        {
            var bitmap = new Bitmap(Image.FromFile(this.path));

            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);

            var pointer = bitmapData.Scan0;

            var bytes = bitmapData.Stride * bitmap.Height;
            var rgbValues = new byte[bytes];

            Marshal.Copy(pointer, rgbValues, 0, bytes);

            this.FloatedContent = new float[3, bitmap.Height, bitmap.Width];
            for (var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < bitmap.Height; ++j)
                {
                    for (var k = 0; k < bitmap.Width; ++k)
                    {
                        this.FloatedContent[i, j, k] = Toolkit.ToFloat(rgbValues[(((j * bitmapData.Stride) + bitmap.Width) * 4) + i]);
                    }
                }
            }

            this.disposeAction = () => bitmap.UnlockBits(bitmapData);

            this.open = true;
        }

        public void Dispose()
        {
            if (this.open)
            {
                this.disposeAction();
                this.FloatedContent = null;
            }
        }
    }
}