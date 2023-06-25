import {OpenAI} from "langchain/llms/openai";
import {ConversationalRetrievalQAChain} from "langchain/chains";
import {TextLoader} from "langchain/document_loaders/fs/text";
import {TokenTextSplitter} from "langchain/text_splitter";
import {Document} from "langchain/document";
import {load} from "langchain/load";
import {OpenAIEmbeddings} from "langchain/embeddings";
import {Chroma} from "langchain/vectorstores";
import {FaissStore} from "langchain/vectorstores/faiss";

class RAG {
    // private field
    #radius;

    constructor(value: String) {
        // You can access private field from constructor
        this.#radius = value;
    }

    private async loadandSplitDoc(): Promise<Document[]> {
        const loader = new TextLoader("src/docs/qa.txt");
        const textSplitter = new TokenTextSplitter({chunkSize: 100, chunkOverlap: 10})
        return await loader.loadAndSplit(textSplitter);
    }

    private newEmbeddings() {
        return new OpenAIEmbeddings;
    }

    public async createAndSaveFaissVectors() {
        process.env.OPENAI_API_KEY = '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'

        const docs = await this.loadandSplitDoc()
        const vectorStore = await FaissStore.fromDocuments(
            docs,
            new OpenAIEmbeddings()
        );
        const directory = "faiss_vectors/";
        await vectorStore.save(directory);

    }

    public async similaritySearch(vectorStoreDirectory: string) {
        process.env.OPENAI_API_KEY = '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'

        const loadedVectorStore = await FaissStore.load(
            vectorStoreDirectory,
            new OpenAIEmbeddings()
        );

        return await loadedVectorStore.similaritySearch("book ticket local currency", 1);
    }
}

const main = async () => {
    let testrag = new RAG('test')
    let search = await testrag.similaritySearch("faiss_vectors/")
    console.log(search);
}

await main();
