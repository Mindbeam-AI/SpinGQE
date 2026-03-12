import torch
from SpinGQE import SpinGQE
from model import GPTConfig
import numpy as np
from hamiltonian import gen_hamiltonian
import os
import pandas as pd
import json
import joblib
import holoviews as hv
import hvplot.pandas
import matplotlib.pyplot as plt
import pennylane as qml
import random


def build_operator_pool(n_qubits, t_values=None):
    if t_values is None:
        # t_values = [np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/8]
        t_values = [np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32]
        t_values += [-t for t in t_values]  # add negatives

    pool = []

    # Two-qubit interactions (Z_i Z_{i+1})
    for i in range(n_qubits - 1):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'ZZ', wires=[i, i + 1]))
            pool.append(qml.PauliRot(t, 'XX', wires=[i, i+1]))
            pool.append(qml.PauliRot(t, 'YY', wires=[i, i+1]))

    # Single-qubit terms (X_i)
    for i in range(n_qubits):
        for t in t_values:
            # pool.append(qml.PauliRot(t, 'X', wires=[i]))
            # pool.append(qml.PauliRot(t, 'Y', wires=i))
            pool.append(qml.PauliRot(t, 'Z', wires=i))

    # for i in range(n_qubits - 2):
    #     for t in t_values:
    #         pool.append(qml.PauliRot(t, 'ZZ', wires=[i, i + 2]))
    #         pool.append(qml.PauliRot(t, 'XX', wires=[i, i+2]))
    #         pool.append(qml.PauliRot(t, 'YY', wires=[i, i+2]))

    return pool


dev = qml.device("default.qubit")


@qml.qnode(dev)
def energy_circuit(seq, ham, init_state, num_qubits):
    qml.BasisState(init_state, wires=range(num_qubits))

    for op in seq:
        qml.Snapshot(measurement=qml.expval(ham))
        qml.apply(op)
    return qml.expval(ham)


energy_circuit = qml.snapshots(energy_circuit)


def get_subsequence_energies(seq, hamiltonian, init_state, num_qubits):
    # Collates the energies of each subsequence for a batch of sequences
    energies = []
    for pool in seq:
        es = energy_circuit(pool, hamiltonian, init_state, num_qubits)
        energies.append(
            [es[k].item() for k in list(range(1, len(pool))) + ["execution_results"]]
        )
    return np.array(energies)


def load_model(load_dir):
    with open(f"{load_dir}/config.json") as f:
        config = json.load(f)

        model = SpinGQE(GPTConfig(
            vocab_size=config["vocab_size"],
            block_size=config["block_size"],
            dropout=config["dropout"],
            bias=config["bias"],
            n_layer=config["n_layer"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
        )).to("cuda")
        load = torch.load(f"{load_dir}/final.pt", map_location="cpu")
        model.load_state_dict(load["model_state_dict"])

        opt = model.configure_optimizers(
            weight_decay=0.01, learning_rate=4e-4, betas=(0.9, 0.95), device_type="auto"
        )

        opt.load_state_dict(load["optimizer_state_dict"])
    return model, opt


def main():
    dir = f"heisenberg"
    os.makedirs(f"{dir}/histo", exist_ok=True)

    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # HYPERPARAMETERS
    ham_label = 1
    num_qubits = 4
    seq_gen = 10 # number of generated circuits per epoch
    seq_len = 12
    temperature = .5
    n_epochs = 700
    n_batches = 10 # number of batches for training, n_batches ideally divides seq_gen
    beta = .3
    load_model = False

    # data = joblib.load(f'./VQE-generated-dataset/data/ground_state/0{num_qubits}qubit/label{ham_label}.jb')
    # grd_E = data["ground_energy"]

    ham = gen_hamiltonian(ham_label, num_qubits, 10, 10)
    grd_E = np.linalg.eigh(qml.matrix(ham))[0][0]
    print(grd_E)
    init_state = [0] * num_qubits
    op_pool = np.array(build_operator_pool(num_qubits), dtype=object)
    op_pool_size = len(op_pool)
    print(op_pool_size)

    if load_model:
        model, opt = load_model("heisenberg_(t=.5,b=1)*")
    else:
        model = SpinGQE(GPTConfig(
            vocab_size=op_pool_size + 1,
            block_size=seq_len,
            dropout=0.2,
            bias=False,
            n_layer=12,
            n_head=8,
            n_embd=512
        )).to("cuda")

        opt = model.configure_optimizers(
            weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.95), device_type="auto"
        )
    
    eval_iter = 50
    current_mae = 10000
    best_checkpoints = [0, 0, 0]
    best_checkpoint_energies = [10000, 10000, 10000]
    max_index = 0
    best_min = 10000
    best_epoch = 0
    
    losses = []
    true_Es_t = []
    pred_Es_t = []
    
    true_Es_g = []
    
    for epoch in range(0, n_epochs+1):
        model.eval()
        with torch.no_grad():
            tokens, _ = model.generate(
                n_sequences=seq_gen,
                max_new_tokens=seq_len,
                temperature=temperature,
                device="cuda"
            )
    
            gen_inds = (tokens[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]
    
            energies = torch.from_numpy(get_subsequence_energies(gen_op_seq, ham, init_state, num_qubits)).to("cuda")
            true_Es_g.append(energies[:, -1].cpu().numpy().reshape(-1, 1))
            train_inds = np.arange(len(tokens))
    
        model.train()
        np.random.shuffle(train_inds)
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        energy_batches = torch.tensor_split(energies[train_inds], n_batches)
        loss_record = 0
        uw_loss_record = 0
    
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad()
            loss, uw_loss = model.calculate_loss(token_batch, energy_batch, beta)
            loss.backward()
            opt.step()
            loss_record += loss.item()
            uw_loss_record += uw_loss.item()
    
        avg_loss = loss_record / n_batches
        avg_uw_loss = uw_loss_record / n_batches
        losses.append(avg_uw_loss) # if you need to generate a plot of the weighted losses, change line to losses.append(avg_loss)
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, {avg_uw_loss}")
    
        if epoch != 0 and epoch % eval_iter == 0:
            # for gpt evaluation
            model.eval()
            with torch.no_grad():
                gen_token_seq, pred_Es = model.generate(
                    n_sequences=100,
                    max_new_tokens=seq_len,
                    temperature=temperature,
                    device="cuda",
                )
    
            pred_Es = pred_Es.detach().cpu().numpy()
    
            gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]
    
            true_Es = get_subsequence_energies(gen_op_seq, ham, init_state, num_qubits)[:, -1].reshape(-1, 1)
    
            mae = np.mean(np.abs(pred_Es - true_Es))
            ave_E = np.mean(true_Es)
            min_E = np.min(true_Es)
    
            pred_Es_t.append(pred_Es)
            true_Es_t.append(true_Es)
    
            # Counts how many times a gate is identical to the one before it
            stutters = 0
            total_gates = 0
            for seq in gen_inds:
                for i in range(1, len(seq)):
                    if seq[i] == seq[i-1]:
                        stutters += 1
                    total_gates += 1
            r = stutters / total_gates
    
            print(f"Iteration: {epoch}, Loss: {losses[-1]}, MAE: {mae}, Ave E: {ave_E}, Min E: {min_E}, Stutter Ratio: {r}")
    
            plt.figure(figsize=(10, 5))
            plt.hist(pred_Es, bins=30, alpha=0.6, label='Predicted Energy')
            plt.hist(true_Es, bins=30, alpha=0.6, label='Measured Energy')
            plt.axvline(min(true_Es), color='red', linestyle='--', label='Min Measured E')
            plt.axvline(sum(pred_Es)/len(pred_Es), color='black', linestyle='--', label='Average Predicted E')
            plt.legend()
            plt.title(f"Energy Distribution @ Epoch {epoch}")
            plt.xlabel("Energy")
            plt.ylabel("Count")
            plt.savefig(f"{dir}/histo/{epoch}")
            plt.close()
    
            if ave_E < best_checkpoint_energies[max_index]:
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,  # optional
                }
                torch.save(save_dict, f"{dir}/checkpoint{epoch}.pt")
    
                best_checkpoints[max_index] = epoch
                best_checkpoint_energies[max_index] = ave_E
                max_index = best_checkpoint_energies.index(max(best_checkpoint_energies))
    
            if min_E < best_min:
                best_min = min_E
                best_epoch = epoch
    
    df_loss = pd.DataFrame(losses)
    df_loss.to_csv(f"{dir}/losses.csv", index=False)
    
    hvplot.extension('matplotlib')
    
    loss_fig = df_loss.hvplot(
        title="Training loss progress", ylabel="loss", xlabel="Training epochs", logy=True
    ).opts(fig_size=600, fontscale=2, aspect=1.2)
    
    hv.save(loss_fig, f"{dir}/loss_fig.png")
    
    pred_Es_t = np.concatenate(pred_Es_t, axis=1)
    true_Es_t = np.concatenate(true_Es_t, axis=1)
    
    df_pred = pd.DataFrame(pred_Es_t, columns=list(range(eval_iter, n_epochs+1, eval_iter)))
    df_true = pd.DataFrame(true_Es_t, columns=list(range(eval_iter, n_epochs+1, eval_iter)))
    
    df_pred.to_csv(f"{dir}/pred_Es_t.csv", index=False)
    df_true.to_csv(f"{dir}/true_Es_t.csv", index=False)
    
    df_true.columns = df_true.columns.astype(int)
    df_pred.columns = df_pred.columns.astype(int)
    
    df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1).reset_index()
    df_trues_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]
    
    df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1).reset_index()
    df_preds_stats.columns = ["Training Iterations", "Ave Pred E", "Min Pred E", "Max Pred E"]
    
    fig = (
        df_trues_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
        df_trues_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
        df_trues_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
    ) * (
        df_preds_stats.hvplot.scatter(x="Training Iterations", y="Ave Pred E", label="Mean Predicted Energies") *
        df_preds_stats.hvplot.line(x="Training Iterations", y="Ave Pred E", alpha=0.5, linewidth=1) *
        df_preds_stats.hvplot.area(x="Training Iterations", y="Min Pred E", y2="Max Pred E", alpha=0.1)
    )
    fig = fig * hv.Curve([[0, grd_E], [n_epochs+1, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4, linestyle="dashed")
    fig = fig.opts(ylabel="Sequence Energies", title="GQE Evaluations", fig_size=600, fontscale=2)
    hv.save(fig, f"{dir}/eval_fig.png")
    
    true_Es_g = np.concatenate(true_Es_g, axis=1)
    df_gen = pd.DataFrame(true_Es_g)
    df_gen.columns = df_gen.columns.astype(int)
    df_gen_stats = pd.concat([df_gen.mean(axis=0), df_gen.min(axis=0), df_gen.max(axis=0)], axis=1).reset_index()
    df_gen_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]
    gen_fig = (
        df_gen_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
        df_gen_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
        df_gen_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
    )
    gen_fig = gen_fig.opts(ylabel="Sequence Energies", title="Generated Training Data", fig_size=600, fontscale=2)
    hv.save(gen_fig, f"{dir}/gen_fig.png")
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }
    torch.save(save_dict, f"{dir}/final.pt")
    
    with open(f"{dir}/config.json", "w") as f:
        json.dump(model.config.__dict__, f, indent=4)
    
    metadata = {
        "num_qubits": num_qubits,
        "ham_label": ham_label,
        "epochs": n_epochs,
        "seq len": seq_len,
        "seq gen": seq_gen,
        "n batches": n_batches,
        "weight beta": beta,
        "lowest energy": best_min,
        "best epoch": best_epoch,
        "best checkpoints": best_checkpoints,
        "final_loss": float(loss.item()),
        "min_mae": float(current_mae),
        "temperature": temperature,
    }
    
    with open(f"{dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()