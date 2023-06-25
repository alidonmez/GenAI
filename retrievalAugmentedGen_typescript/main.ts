import {TextLoader} from "langchain/document_loaders/fs/text";
import {TokenTextSplitter} from "langchain/text_splitter";
import {Document} from "langchain/document";
import {OpenAIEmbeddings} from "langchain/embeddings";
import {FaissStore} from "langchain/vectorstores/faiss";

class RetrievalAugmentedGeneration {
    constructor() {
    }

    private async loadAndSplitDoc(): Promise<Document[]> {
        const loader = new TextLoader("src/docs/qa.txt");
        const textSplitter = new TokenTextSplitter({ chunkSize: 100, chunkOverlap: 10 });
        return await loader.loadAndSplit(textSplitter);
    }

    private newEmbeddings(): OpenAIEmbeddings {
        return new OpenAIEmbeddings();
    }

    public async createAndSaveFaissVectors(vectorStoreDirectory: string): Promise<void> {
        const docs = await this.loadAndSplitDoc();
        const vectorStore = await FaissStore.fromDocuments(docs, this.newEmbeddings());
        await vectorStore.save(vectorStoreDirectory);
    }

    public async similaritySearch(question: string, vectorStoreDirectory: string): Promise<any> {
        const loadedVectorStore = await FaissStore.load(vectorStoreDirectory, this.newEmbeddings());
        return await loadedVectorStore.similaritySearch(question, 2);
    }
}

const main = async (): Promise<void> => {
    const retrievalAugGen = new RetrievalAugmentedGeneration();
    const search = await retrievalAugGen.similaritySearch("enter question here","faiss_vectors/");
    console.log(search);
};

await main();
