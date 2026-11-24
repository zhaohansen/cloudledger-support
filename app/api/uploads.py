from fastapi import HTTPException, UploadFile

from app.config import get_settings

settings = get_settings()


async def read_upload(file: UploadFile, label: str = "文件") -> bytes:
    limit = settings.upload_max_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"{label}不能超过 {settings.upload_max_mb} MB")
    if not content:
        raise HTTPException(status_code=422, detail=f"{label}为空")
    return content
