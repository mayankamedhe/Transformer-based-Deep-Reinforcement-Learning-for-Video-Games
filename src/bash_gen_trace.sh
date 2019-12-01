for i in {1..10}
do
    echo "training DQN trace $i ..."
    python train_DQN.py > ../out/trace_DQN_"$i".txt
done

for i in {1..10}
do
    echo "training DRQN trace $i ..."
    python train_DRQN.py > ../out/trace_DRQN_"$i".txt
done

for i in {1..10}
do
    echo "training DTQN trace $i ..."
    python train_DTQN.py > ../out/trace_DTQN_"$i".txt
done