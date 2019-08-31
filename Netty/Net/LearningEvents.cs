using System;
using System.Threading.Tasks;

namespace Netty.Net
{
    public class LearningEvents
    {
        private Task task;

        public event EventHandler<AllDoneArgs> AllDone;

        public event EventHandler<EpochDoneArgs> EpochDone;

        public event EventHandler<EpochProgressUpdateArgs> EpochProgressUpdate;

        public async Task InvokeAllDone(object sender, int totalEpochs, float finalError)
        {
            AllDone?.Invoke(sender, new AllDoneArgs
            {
                TotalEpochs = totalEpochs,
                FinalError = finalError
            });
        }

        public async Task InvokeEpochDone(object sender, int doneEpochs, int totalEpochs, float currentError)
        {
            if (task != null)
            {
                await task;
                task.Dispose();
            }

            task = Task.Run(() => 
                EpochDone?.Invoke(sender, new EpochDoneArgs
                {
                    DoneEpochs = doneEpochs,
                    TotalEpochs = totalEpochs,
                    CurrentError = currentError
                })
            );
        }

        public async Task InvokeEpochProgressUpdate(object sender, int samplesDone, int samplesTotal)
        {
            if (task != null)
            {
                if (!task.IsCompleted)
                {
                    return;
                }

                task.Dispose();
            }

            task = Task.Run(() =>
                EpochProgressUpdate?.Invoke(sender, new EpochProgressUpdateArgs
                {
                    SamplesDone = samplesDone,
                    SamplesTotal = samplesTotal
                })
            );
        }
    }
}
