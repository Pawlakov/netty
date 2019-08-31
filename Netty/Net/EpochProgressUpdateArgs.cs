using System;

namespace Netty.Net
{
    public class EpochProgressUpdateArgs : EventArgs
    {
        public int SamplesDone { get; set; }

        public int SamplesTotal { get; set; }
    }
}