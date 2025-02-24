import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embeddings(embeddings, text, which_lvls=[1,1,1,1], filename="vis_embs.png", annotate=True):
    do_subwords = bool(which_lvls[0])
    do_words    = bool(which_lvls[1]) and len(embeddings) >= 2
    do_phrase2s = bool(which_lvls[2]) and len(embeddings) >= 3
    do_phrase3s = bool(which_lvls[3]) and len(embeddings) >= 4

    # Extract the embeddings from the nested list
    subword_embs, word_embs, phrase2_embs, phrase3_embs = embeddings
    
    if isinstance(subword_embs, list):
        # Normalize each subword embedding array by magnitude
        for i in range(len(subword_embs)):
            subword_embs[i] /= np.linalg.norm(subword_embs[i], axis=1)[:, np.newaxis]
        
        # Flatten each subword embedding array for PCA or t-SNE
        subword_embs = [np.reshape(subword_embs[i], (-1, subword_embs[i].shape[-1])) for i in range(len(subword_embs))]
    else:
        # Normalize subword_embs (if not a list) and flatten
        subword_embs /= np.linalg.norm(subword_embs, axis=1)[:, np.newaxis]
        subword_embs = [np.reshape(subword_embs, (-1, subword_embs.shape[-1]))]

    word_embs /= np.linalg.norm(word_embs, axis=1)[:, np.newaxis]
    phrase2_embs /= np.linalg.norm(phrase2_embs, axis=1)[:, np.newaxis]
    phrase3_embs /= np.linalg.norm(phrase3_embs, axis=1)[:, np.newaxis]

    # Flatten the other embeddings for PCA or t-SNE
    word_embs = np.reshape(word_embs, (-1, word_embs.shape[-1]))
    phrase2_embs = np.reshape(phrase2_embs, (-1, phrase2_embs.shape[-1]))
    phrase3_embs = np.reshape(phrase3_embs, (-1, phrase3_embs.shape[-1]))
    
    # Combine embeddings based on which levels to include
    embeddings_to_reduce = []
    if do_subwords:
        embeddings_to_reduce.extend(subword_embs)  # Extend with each subword level
    if do_words:
        embeddings_to_reduce.append(word_embs)
    if do_phrase2s:
        embeddings_to_reduce.append(phrase2_embs)
    if do_phrase3s:
        embeddings_to_reduce.append(phrase3_embs)
        
    all_embs = np.vstack(embeddings_to_reduce)
    
    # Reduce dimensions to 2 using t-SNE
    reduced_embs = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(all_embs)
    
    # Calculate the split indices for different types of embeddings
    current_index = 0
    subword_indices = []
    if do_subwords:
        for i, emb in enumerate(subword_embs):
            subword_len = len(emb)
            subword_indices.append(np.arange(current_index, current_index + subword_len))
            current_index += subword_len
    if do_words:
        word_len = len(word_embs)
        word_indices = np.arange(current_index, current_index + word_len)
        current_index += word_len
    if do_phrase2s:
        phrase2_len = len(phrase2_embs)
        phrase2_indices = np.arange(current_index, current_index + phrase2_len)
        current_index += phrase2_len
    if do_phrase3s:
        phrase3_len = len(phrase3_embs)
        phrase3_indices = np.arange(current_index, current_index + phrase3_len)
        current_index += phrase3_len
    
    # Extract words, phrases2, and phrases3 from the text
    words = text.split()
    phrase2s = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
    phrase3s = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]

    # Plot the embeddings
    plt.figure(figsize=(20, 20))
    
    # Subword embeddings: different shades of gray
    if do_subwords:
        num_levels = len(subword_embs)
        colors = plt.cm.gray(np.linspace(0.9, 0.1, num_levels))  # Generate shades of gray in reverse order
        for i, indices in enumerate(subword_indices):
            if i == 1:
                plt.scatter(reduced_embs[indices, 0], reduced_embs[indices, 1],
                            s=i, color=colors[i], label=f'Subword Level', alpha=0.5)
            else:
                plt.scatter(reduced_embs[indices, 0], reduced_embs[indices, 1],
                            s=i, color=colors[i], alpha=0.5)
    
    # Word embeddings: bigger red dots
    if do_words:
        plt.scatter(reduced_embs[word_indices, 0], reduced_embs[word_indices, 1], 
                    s=15, c='red', label='Word Embeddings', alpha=0.5)
    
    # Phrase2 embeddings: biggest blue dots
    if do_phrase2s:
        plt.scatter(reduced_embs[phrase2_indices, 0], reduced_embs[phrase2_indices, 1], 
                    s=20, c='blue', label='Phrase (2-word) Embeddings', alpha=0.5)
    
    # Phrase3 embeddings: biggest green dots
    if do_phrase3s:
        plt.scatter(reduced_embs[phrase3_indices, 0], reduced_embs[phrase3_indices, 1], 
                    s=25, c='green', label='Phrase (3-word) Embeddings', alpha=0.5)
    
    # Connect each word with its corresponding phrase2
    if do_words and do_phrase2s:
        for i in range(len(word_embs) - 1):
            plt.plot([reduced_embs[word_indices[i], 0], reduced_embs[phrase2_indices[i], 0]], 
                     [reduced_embs[word_indices[i], 1], reduced_embs[phrase2_indices[i], 1]], c='red', linewidth=0.1, alpha=0.2)
            plt.plot([reduced_embs[word_indices[i + 1], 0], reduced_embs[phrase2_indices[i], 0]], 
                     [reduced_embs[word_indices[i + 1], 1], reduced_embs[phrase2_indices[i], 1]], c='red', linewidth=0.1, alpha=0.2)
    
    # Connect each phrase2 with its corresponding phrase3
    if do_phrase2s and do_phrase3s:
        for i in range(len(phrase2_embs) - 1):
            plt.plot([reduced_embs[phrase2_indices[i], 0], reduced_embs[phrase3_indices[i], 0]], 
                     [reduced_embs[phrase2_indices[i], 1], reduced_embs[phrase3_indices[i], 1]], c='blue', linewidth=0.1, alpha=0.2)
            plt.plot([reduced_embs[phrase2_indices[i + 1], 0], reduced_embs[phrase3_indices[i], 0]], 
                     [reduced_embs[phrase2_indices[i + 1], 1], reduced_embs[phrase3_indices[i], 1]], c='blue', linewidth=0.1, alpha=0.2)

    # Annotate word embeddings
    if annotate and do_words:
        for i, word in enumerate(words):
            plt.annotate("    " + word, (reduced_embs[word_indices[i], 0], reduced_embs[word_indices[i], 1]), fontsize=4, color='black', alpha=0.5)
    
    # Annotate phrase2 embeddings
    if annotate and do_phrase2s:
        for i, phrase in enumerate(phrase2s):
            plt.annotate("    " + phrase, (reduced_embs[phrase2_indices[i], 0], reduced_embs[phrase2_indices[i], 1]), fontsize=4, color='black', alpha=0.5)
    
    # Annotate phrase3 embeddings
    if annotate and do_phrase3s:
        for i, phrase in enumerate(phrase3s):
            plt.annotate("    " + phrase, (reduced_embs[phrase3_indices[i], 0], reduced_embs[phrase3_indices[i], 1]), fontsize=4, color='black', alpha=0.5)
    
    plt.legend()
    plt.title('Embeddings Visualized')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(filename, bbox_inches='tight', dpi=300)




long_text = "in continental philosophy ( particularly phenomenology and existentialism ) , there is much greater tolerance of ambiguity , as it is generally seen as an integral part of the human condition . martin heidegger argued that the relation between the subject and object is ambiguous , as is the relation of mind and body , and part and whole . [ 3 ] in heidegger ' s phenomenology , dasein is always in a meaningful world , but there is always an underlying background for every instance of signification . thus , although some things may be certain , they have little to do with dasein ' s sense of care and existential anxiety , e.g. , in the face of death . in calling his work being and nothingness an \" essay in phenomenological ontology \" jean-paul sartre follows heidegger in defining the human essence as ambiguous , or relating fundamentally to such ambiguity . simone de beauvoir tries to base an ethics on heidegger ' s and sartre ' s writings ( the ethics of ambiguity ) , where she highlights the need to grapple with ambiguity : \" as long as philosophers and they [ men ] have thought , most of them have tried to mask it ... and the ethics which they have proposed to their disciples have always pursued the same goal . it has been a matter of eliminating the ambiguity by making oneself pure inwardness or pure externality , by escaping from the sensible world or being engulfed by it , by yielding to eternity or enclosing oneself in the pure moment . \" ethics can not be based on the authoritative certainty given by mathematics and logic , or prescribed directly from the empirical findings of science . she states : \" since we do not succeed in fleeing it , let us , therefore , try to look the truth in the face . let us try to assume our fundamental ambiguity . it is in the knowledge of the genuine conditions of our life that we must draw our strength to live and our reason for acting \" . other continental philosophers suggest that concepts such as life , nature , and sex are ambiguous . corey anton has argued that we can not be certain what is separate from or unified with something else : language , he asserts , divides what is not , in fact , separate . following ernest becker , he argues that the desire to ' authoritatively disambiguate ' the world and existence has led to numerous ideologies and historical events such as genocide . on this basis , he argues that ethics must focus on ' dialectically integrating opposites ' and balancing tension , rather than seeking a priori validation or certainty . like the existentialists and phenomenologists , he sees the ambiguity of life as the basis of creativity ."
numbers = ' '.join([str(i) for i in range(100)]) + " one two three four five six seven eight nine" + " " + ' '.join([str(year) for year in range(1950, 2025)])
colors = "red blue green yellow orange purple pink brown black white gray cyan magenta maroon navy teal olive lime indigo violet turquoise gold silver bronze beige coral salmon lavender tan mint peach ivory azure burgundy charcoal aquamarine amber cerulean fuchsia sepia"
names = "karlo ines james john robert michael william david richard joseph charles thomas mary patricia jennifer linda elizabeth barbara susan jessica sarah karen alexander maximilian oliver lucas leon finn noah emil jan theo emma sophia mia emilia anna sofie clara eva lea lina"
countries = "usa united states canada mexico brazil argentina united kingdom britain france germany italy spain russia china japan south korea india australia new zealand south africa egypt nigeria kenya saudi arabia turkey iran indonesia thailand vietnam malaysia philippines singapore"
cities = "washington ottawa mexico city brasilia buenos aires london paris berlin rome madrid moscow beijing tokyo seoul new delhi canberra wellington pretoria cairo abuja nairobi riyadh ankara tehran jakarta bangkok hanoi kuala lumpur manila singapore"
opposites = "good bad white black red blue hot cold girl boy high low"
chars = "a b c d e f g h i j k l m n o p r s š t u v z ž x y z ! \" # $ % & / ( ) = ? * + '"
asdfg = "žš čćžš fguj htezs wetheq wneriguw ilgbwwh erzherzjemter jrzujerthuoh wterhehdtaj wqergbiwq wergiuqwbergoiwzuer werbgiwuzebgoqwierzgb qerbgizqebroqzerbg arhadfgauswvfka rigzabfvuavsrizab asigzabsflizasazrkjdsf jdsrtjsdrtjewrtmrtuzjsjnu ohwetgouwregočiuh nornbčwoh wtčeguhwretguwbtrgliubwr kvwerufzvkrefv berzifoqizrevfue boerzvqoueezrvcwuegfoquez boqew8zvowqevgvoizergfoihfoiazer izwebrfoizqgerfloi9uqgerpg98onqewroiuwerg ouwherfouqwerflizbqeroizgwertizrtgtrgiug"
text = long_text + " " + numbers + " " + colors + " " + names + " " + countries + " " + cities + " " + opposites + " " + chars + " " + asdfg


"""
embs = text_emb_lvl(model, text, 2, multilvl=True)
plot_embeddings(embs, text, which_lvls=[1,1,0,0], filename="vis_embs_sw.png", annotate=True)
plot_embeddings(embs, text, which_lvls=[0,1,1,1], filename="vis_embs_wp.png", annotate=True)
"""