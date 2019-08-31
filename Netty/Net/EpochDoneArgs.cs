using System;

namespace Netty.Net
{
    public class EpochDoneArgs : EventArgs
    {
        public int DoneEpochs { get; set; }

        public int TotalEpochs { get; set; }

        public float CurrentError { get; set; }
    }
}