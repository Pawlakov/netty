namespace ClickbaitGenerator.NeuralNet.Model
{
    public sealed class Ref<T>
    {
        public T Value { get; set; }

        public Ref(T weight)
        {
            this.Value = weight;
        }

    }
}
