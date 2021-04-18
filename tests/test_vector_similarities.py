from tests.helpers import cosine_similarity


def test_similarities_are_reasonable(learned_embeddings):
    vectors = learned_embeddings
    assert cosine_similarity(vectors["suit"], vectors["tie"]) > cosine_similarity(
        vectors["suit"], vectors["high_heels"]
    )
    assert cosine_similarity(
        vectors["suit"], vectors["ball_dress"]
    ) > cosine_similarity(vectors["suit"], vectors["high_heels"])
    assert cosine_similarity(
        vectors["sun_dress"], vectors["ball_dress"]
    ) > cosine_similarity(vectors["suit"], vectors["winter_gloves"])
    assert cosine_similarity(
        vectors["winter_gloves"], vectors["winter_gloves"]
    ) > cosine_similarity(vectors["suit"], vectors["winter_gloves"])


def test_training_improved_similarities(learned_embeddings, pre_learned_embeddings):
    def should_get_more_similar_with_training(item1: str, item2: str) -> None:
        vectors = learned_embeddings
        vectors2 = pre_learned_embeddings
        assert cosine_similarity(vectors[item1], vectors[item2]) > cosine_similarity(
            vectors2[item1], vectors2[item2]
        )

    should_get_more_similar_with_training("sun_dress", "ball_dress")
    should_get_more_similar_with_training("winter_gloves", "winter_hat")
    should_get_more_similar_with_training("suit", "tie")
