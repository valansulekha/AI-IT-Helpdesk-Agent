import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# RAG CONFIGURATION
# ============================================================

KNOWLEDGE_FILE = "it_knowledge.pkl"
INDEX_FILE = "it_knowledge.index"

# Sentence Transformer model
# This converts IT knowledge into numerical embeddings.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================
# IT KNOWLEDGE BASE
# ============================================================

knowledge_base = [

    {
        "title": "Network Connectivity Troubleshooting",
        "category": "Network Issue",
        "content": """
        If a computer cannot connect to the internet or Wi-Fi,
        first check whether Wi-Fi or Ethernet is enabled.
        Restart the router or network device if necessary.
        Forget and reconnect to the Wi-Fi network.
        Restart the computer and network adapter.
        Check the IP address and network configuration.
        If other devices also cannot connect, the problem may
        be related to the router or internet service provider.
        """
    },

    {
        "title": "Slow Computer and System Freezing",
        "category": "Performance Issue",
        "content": """
        A slow computer or frequent freezing can be caused by
        high CPU or RAM usage, insufficient storage, too many
        startup applications, background processes, malware,
        or outdated software.
        Close unnecessary applications and check CPU and memory
        usage using Task Manager.
        Remove unnecessary startup programs.
        Free up disk space and restart the computer.
        Install important operating system and software updates.
        """
    },

    {
        "title": "Printer Troubleshooting",
        "category": "Printer Issue",
        "content": """
        If a printer is not printing, check that the printer is
        powered on and connected correctly.
        Check paper, ink or toner.
        Make sure the correct printer is selected.
        Clear stuck print jobs from the print queue.
        Restart the printer and computer.
        Check whether the printer driver is installed correctly.
        """
    },

    {
        "title": "Password and Account Problems",
        "category": "Account Issue",
        "content": """
        If a user cannot log in or has forgotten a password,
        verify the username and password.
        Check whether Caps Lock is enabled.
        Use the organization's password reset procedure.
        If the account is locked, contact the IT administrator.
        Never share passwords with other users.
        """
    },

    {
        "title": "Phishing and Security Incidents",
        "category": "Security Issue",
        "content": """
        Suspicious emails, phishing links, malware warnings,
        and unauthorized access should be treated as security
        incidents.
        Do not click suspicious links or open unknown attachments.
        Do not provide passwords or sensitive information.
        Report suspicious messages to the IT or security team.
        Disconnect the affected device from the network if malware
        infection is suspected and follow the organization's
        security incident procedure.
        """
    },

    {
        "title": "Hardware Troubleshooting",
        "category": "Hardware Issue",
        "content": """
        Hardware problems may include battery failures,
        keyboard problems, mouse problems, display issues,
        overheating, or devices that do not power on.
        Check power connections, cables, ports, and external
        devices.
        Restart the computer if possible.
        Check for overheating and unusual hardware sounds.
        If the hardware remains faulty, contact IT support.
        """
    },

    {
        "title": "Email Troubleshooting",
        "category": "Email Issue",
        "content": """
        If email messages cannot be sent or received,
        check internet connectivity first.
        Verify email account credentials and server settings.
        Check mailbox storage limits.
        Restart the email application.
        Check whether the email service is experiencing an outage.
        Contact IT support if the problem continues.
        """
    },

    {
        "title": "Software Application Problems",
        "category": "Software Issue",
        "content": """
        If an application crashes, freezes, or does not open,
        restart the application.
        Restart the computer if necessary.
        Check for application updates.
        Verify that the computer meets the software requirements.
        Reinstall the application if the problem continues.
        Check application error messages and logs when available.
        """
    },

    {
        "title": "Access and Permission Problems",
        "category": "Access/Permission Issue",
        "content": """
        If a user cannot access a shared folder, application,
        file, or system resource, verify the user's account.
        Check whether the user has the required permissions.
        Confirm that the resource is available.
        Contact the administrator when additional permissions
        are required.
        Users should not attempt to bypass access controls.
        """
    },

    {
        "title": "Database Connection Problems",
        "category": "Database Issue",
        "content": """
        If an application cannot connect to a database,
        check network connectivity and database server status.
        Verify the database hostname, port, username, and password.
        Check whether the database service is running.
        Review application or database error logs.
        Contact the database administrator if the connection
        problem continues.
        """
    }
]

# ============================================================
# CREATE RAG INDEX
# ============================================================

def build_knowledge_base():

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []

    for item in knowledge_base:

        text = (
            item["title"]
            + "\n"
            + item["category"]
            + "\n"
            + item["content"]
        )

        documents.append(text)

    print("Creating embeddings...")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    # FAISS similarity search index
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)

    # Save knowledge documents
    with open(KNOWLEDGE_FILE, "wb") as file:
        pickle.dump(knowledge_base, file)

    print()
    print("=" * 60)
    print("RAG KNOWLEDGE BASE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Embedding dimension: {dimension}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Knowledge file: {KNOWLEDGE_FILE}")
    print("=" * 60)


# ============================================================
# SEARCH KNOWLEDGE BASE
# ============================================================

def search_knowledge(query, top_k=3):

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            "RAG index not found. Run: python knowledge_base.py"
        )

    if not os.path.exists(KNOWLEDGE_FILE):
        raise FileNotFoundError(
            "Knowledge file not found. Run: python knowledge_base.py"
        )

    model = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(INDEX_FILE)

    with open(KNOWLEDGE_FILE, "rb") as file:
        documents = pickle.load(file)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):

        if index_number == -1:
            continue

        result = documents[index_number].copy()

        result["similarity_score"] = float(score)

        results.append(result)

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_knowledge_base()

    print()
    print("Testing RAG search...")
    print()

    test_query = "My laptop is very slow and keeps freezing"

    results = search_knowledge(
        test_query,
        top_k=3
    )

    print(f"Query: {test_query}")
    print()

    for number, result in enumerate(results, start=1):

        print(f"Result {number}")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(
            f"Similarity: "
            f"{result['similarity_score']:.3f}"
        )
        print()