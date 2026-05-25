from dataclasses import dataclass
from typing import TypeAlias

ResultKey: TypeAlias = tuple[str, str, str]


def canonical_result_model(model: str, provider: str) -> str:
    if provider == "mudler" and model.endswith("-APEX"):
        return model[:-5]
    return model


def render_model_tag(repo: str, quant: str) -> str:
    return f"{repo}:{quant}"


def display_name_from_repo(repo: str) -> str:
    name = repo.split("/")[-1]
    for prefix in (
        "google_",
        "Qwen_",
        "qwen_",
        "zai-org_",
        "mistralai_",
        "nvidia_",
        "NVIDIA-",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith("-APEX-GGUF"):
        name = name[:-10]
    if name.endswith("-GGUF"):
        name = name[:-5]
    if name.endswith("-it"):
        name = name[:-3]
    return name


@dataclass(frozen=True)
class ModelIdentity:
    repo: str
    quant: str
    provider: str

    @classmethod
    def from_tag(cls, tag: str, *, require_quant: bool = True) -> "ModelIdentity":
        repo, separator, quant = tag.partition(":")
        if require_quant and not separator:
            raise ValueError(f"model tag requires a quant suffix: {tag!r}")
        provider = repo.split("/", 1)[0]
        return cls(repo=repo, quant=quant if separator else "", provider=provider)

    @classmethod
    def from_repo_quant(cls, repo: str, quant: str) -> "ModelIdentity":
        provider = repo.split("/", 1)[0]
        return cls(repo=repo, quant=quant, provider=provider)

    @property
    def display_name(self) -> str:
        return display_name_from_repo(self.repo)

    @property
    def result_key(self) -> ResultKey:
        return (self.display_name, self.quant, self.provider)


def identity_from_tag(tag: str, *, require_quant: bool = True) -> ModelIdentity:
    return ModelIdentity.from_tag(tag, require_quant=require_quant)


def result_key_from_tag(tag: str, *, require_quant: bool = True) -> ResultKey:
    return identity_from_tag(tag, require_quant=require_quant).result_key


def result_key_from_parts(repo: str, quant: str) -> ResultKey:
    return ModelIdentity.from_repo_quant(repo, quant).result_key
