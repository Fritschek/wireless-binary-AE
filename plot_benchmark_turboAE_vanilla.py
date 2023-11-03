import matplotlib.pyplot as plt
import pickle
import seaborn as sns


def plot():
    with open('data_turboae_400epochs_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    with open("data_turboae_300epochs.pkl", 'rb') as f:
        data2 = pickle.load(f)
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    (BER, BLER, SNR, test_loss, test_ber) = data
    (test_loss2, test_ber2, loss_Enc, loss_dec) = data2
    
    colors = sns.color_palette("deep")
        
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # Plot SNR vs BER on the top-left subplot
    sns.lineplot(x=SNR, y=BER, ax=axs[0, 0], color=colors[0])
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid(True, which="both",ls="--",  c='0.65')
    axs[0, 0].set_title('SNR vs BER')
    axs[0, 0].set_xlabel('SNR')  

    
    # Plot SNR vs BLER on the top-right subplot
    sns.lineplot(x=SNR, y=BLER, ax=axs[0, 1], color=colors[0])
    axs[0, 1].plot(SNR, BLER, label='BLER')
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True, which="both",ls="--",  c='0.65')
    axs[0, 1].set_title('SNR vs BLER')
    axs[0, 1].set_xlabel('SNR') 

    
    # Plot test_loss on the bottom-left subplot
    sns.lineplot(x=list(range(len(test_loss))), y=test_loss, ax=axs[1, 0], color=colors[0], label='run1')
    sns.lineplot(x=list(range(len(test_loss2))), y=test_loss2, ax=axs[1, 0], color=colors[1], label = 'run2')
    axs[1, 0].plot(test_loss, label='Test Loss')
    axs[1, 0].grid(True, which="both",ls="--",  c='0.65')
    axs[1, 0].set_title('Test Loss')
    axs[1, 0].set_xlabel('Epochs')  

    
    # Plot test_ber on the bottom-right subplot
    sns.lineplot(x=list(range(len(test_ber))), y=test_ber, ax=axs[1, 1], color=colors[0], label='run1')
    sns.lineplot(x=list(range(len(test_ber2))), y=test_ber2, ax=axs[1, 1], color=colors[1], label = 'run2')
    axs[1, 1].plot(test_ber, label='Test BER')
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True, which="both",ls="--",  c='0.65')
    axs[1, 1].set_title('Test BER')
    axs[1, 1].set_xlabel('Epochs')
    #
    # encoder decoder loss run2
    sns.lineplot(x=list(range(len(loss_Enc))), y=loss_Enc, ax=axs[1, 2], color=colors[0], label='enc')
    sns.lineplot(x=list(range(len(loss_dec))), y=loss_dec, ax=axs[1, 2], color=colors[1], label = 'dec')
    axs[1, 2].grid(True, which="both",ls="--",  c='0.65')
    axs[1, 2].set_title('Test loss')
    axs[1, 2].set_xlabel('Epochs') 

    plt.tight_layout()
    #plt.show()
    
    plt.savefig('plot.png', format='png', dpi=300)

