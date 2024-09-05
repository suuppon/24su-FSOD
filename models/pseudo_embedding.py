import torch
import torch.nn as nn
import torch.nn.functional as F

class PseudoEmbedding(nn.Module):
    def __init__(self, pseudo_num_classes, embedding_dim):
        """
        Initializes the PseudoEmbedding module.

        Args:
            num_classes (int): Number of pseudo-classes.
            embedding_dim (int): Dimension of the embeddings.
        """
        super(PseudoEmbedding, self).__init__()
        # Initialize pseudo-class embeddings from a normal distribution
        self.embeddings = nn.Parameter(torch.empty(pseudo_num_classes, embedding_dim))
        nn.init.normal_(self.embeddings, mean=0.0, std=1.0)  # Normal distribution initialization

    def forward(self, template_features, template_labels):
        """
        Forward method to add pseudo-class embeddings to template features.

        Args:
            template_features (torch.Tensor): Template features from the backbone. Shape: (batch_size, num_templates, embedding_dim)
            template_labels (torch.Tensor): Class labels of the templates. Shape: (batch_size, num_templates)

        Returns:
            torch.Tensor: Modified template features stamped with pseudo-class embeddings.
        """
        # template_features: (batch_size, num_templates, embedding_dim)
        # template_labels: (batch_size, num_templates)

        batch_size, num_templates, embedding_dim = template_features.shape
        # Check if the number of classes exceeds the number of pseudo-classes
        assert torch.max(template_labels) < self.embeddings.size(0), "Number of classes exceeds the number of pseudo-classes."
        
        # Randomly permute pseudo-class embeddings each forward pass
        # permuted_embeddings: (num_classes, embedding_dim) after shuffling
        permuted_embeddings = self.embeddings[torch.randperm(self.embeddings.size(0))]

        # Map templates to their corresponding pseudo-class embeddings using template labels
        # pseudo_class_embeddings: (batch_size, num_templates, embedding_dim)
        pseudo_class_embeddings = permuted_embeddings[template_labels]

        # Add the pseudo-class embeddings to the template features
        # stamped_features: (batch_size, num_templates, embedding_dim)
        stamped_features = template_features + pseudo_class_embeddings

        return stamped_features  # Output shape: (batch_size, num_templates, embedding_dim)

# Example usage
def main():
    # Define parameters
    batch_size = 4
    num_templates = 3
    embedding_dim = 256
    num_classes = 10

    # Initialize PseudoEmbedding module
    pseudo_embedding = PseudoEmbedding(num_classes=num_classes, embedding_dim=embedding_dim)

    # Example template features from a CNN backbone (randomly initialized for demonstration)
    template_features = torch.randn(batch_size, num_templates, embedding_dim)  # Shape: (batch_size, num_templates, embedding_dim)

    # Example template labels (random classes for each template in the batch)
    template_labels = torch.randint(0, num_classes, (batch_size, num_templates))  # Shape: (batch_size, num_templates)

    # Get stamped template features with random pseudo-class embeddings
    stamped_features = pseudo_embedding(template_features, template_labels)

    # Output the result
    print("Stamped Features Shape:", stamped_features.shape)
    print("Stamped Features:", stamped_features)

if __name__ == "__main__":
    main()