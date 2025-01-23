import arxiv
import os
import time
from urllib.request import urlretrieve

def download_arxiv_papers(search_query, num_papers=199, output_dir="arxiv_papers"):
    """
    Download papers from arXiv based on a search query.
    
    Args:
        search_query (str): Search query for arXiv papers
        num_papers (int): Number of papers to download (default: 199)
        output_dir (str): Directory to save the papers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the search client
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    print(f"Downloading {num_papers} papers related to '{search_query}'...")
    
    # Download papers
    for i, result in enumerate(client.results(search), 1):
        # Create a clean filename from the title
        clean_title = "".join(c if c.isalnum() else "_" for c in result.title)
        filename = f"{clean_title[:100]}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        try:
            print(f"[{i}/{num_papers}] Downloading: {result.title}")
            urlretrieve(result.pdf_url, filepath)
            
            # Be nice to arXiv's servers with a small delay
            time.sleep(3)
            
        except Exception as e:
            print(f"Error downloading {result.title}: {str(e)}")
            continue

    print("\nDownload complete!")

if __name__ == "__main__":
    # Example usage: downloading papers related to quantum computing
    search_query = "quantum computing"
    download_arxiv_papers(search_query)