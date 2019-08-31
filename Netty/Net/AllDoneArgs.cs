using System;

namespace Netty.Net
{
    public class AllDoneArgs : EventArgs
    {
        public int TotalEpochs { get; set; }

        public float FinalError { get; set; }
    }
}