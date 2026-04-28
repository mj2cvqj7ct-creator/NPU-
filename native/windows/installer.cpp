#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <string>

namespace {

std::wstring LastErrorMessage(DWORD error) {
    wchar_t* buffer = nullptr;
    FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error,
        0,
        reinterpret_cast<LPWSTR>(&buffer),
        0,
        nullptr
    );
    std::wstring message = buffer ? buffer : L"unknown error";
    if (buffer) {
        LocalFree(buffer);
    }
    return message;
}

std::wstring DirectoryOf(const std::wstring& path) {
    const size_t pos = path.find_last_of(L"\\/");
    return pos == std::wstring::npos ? L"." : path.substr(0, pos);
}

std::wstring CurrentExePath() {
    wchar_t buffer[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return buffer;
}

std::wstring KnownFolder(REFKNOWNFOLDERID folderId) {
    PWSTR path = nullptr;
    HRESULT hr = SHGetKnownFolderPath(folderId, 0, nullptr, &path);
    if (FAILED(hr)) {
        throw std::wstring(L"SHGetKnownFolderPath failed");
    }
    std::wstring result = path;
    CoTaskMemFree(path);
    return result;
}

void EnsureDirectory(const std::wstring& path) {
    if (CreateDirectoryW(path.c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS) {
        return;
    }
    throw L"CreateDirectory failed: " + LastErrorMessage(GetLastError());
}

void CreateDesktopShortcut(const std::wstring& targetPath, const std::wstring& installDir) {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool initialized = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        throw std::wstring(L"COM initialization failed");
    }

    IShellLinkW* link = nullptr;
    hr = CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW, reinterpret_cast<void**>(&link));
    if (FAILED(hr)) {
        if (initialized) {
            CoUninitialize();
        }
        throw std::wstring(L"Could not create ShellLink COM object");
    }

    link->SetPath(targetPath.c_str());
    link->SetWorkingDirectory(installDir.c_str());
    link->SetDescription(L"Native NPU streaming music enhancer");
    link->SetIconLocation(targetPath.c_str(), 0);

    IPersistFile* file = nullptr;
    hr = link->QueryInterface(IID_IPersistFile, reinterpret_cast<void**>(&file));
    if (FAILED(hr)) {
        link->Release();
        if (initialized) {
            CoUninitialize();
        }
        throw std::wstring(L"Could not access shortcut persistence");
    }

    const std::wstring shortcutPath = KnownFolder(FOLDERID_Desktop) + L"\\NPU Streaming Music Enhancer.lnk";
    hr = file->Save(shortcutPath.c_str(), TRUE);
    file->Release();
    link->Release();
    if (initialized) {
        CoUninitialize();
    }
    if (FAILED(hr)) {
        throw std::wstring(L"Could not save desktop shortcut");
    }
}

void Install() {
    const std::wstring installerPath = CurrentExePath();
    const std::wstring sourceExe = DirectoryOf(installerPath) + L"\\NPUStreamingMusicEnhancer.exe";
    const DWORD attrs = GetFileAttributesW(sourceExe.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES || (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        throw L"NPUStreamingMusicEnhancer.exe must be in the same folder as this installer.";
    }

    const std::wstring installDir = KnownFolder(FOLDERID_LocalAppData) + L"\\NPUStreamingMusicEnhancer";
    EnsureDirectory(installDir);
    const std::wstring targetExe = installDir + L"\\NPUStreamingMusicEnhancer.exe";
    if (!CopyFileW(sourceExe.c_str(), targetExe.c_str(), FALSE)) {
        throw L"CopyFile failed: " + LastErrorMessage(GetLastError());
    }
    CreateDesktopShortcut(targetExe, installDir);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    try {
        Install();
        MessageBoxW(
            nullptr,
            L"Installed NPU Streaming Music Enhancer and created a Windows desktop shortcut.",
            L"NPU Streaming Music Enhancer Installer",
            MB_OK | MB_ICONINFORMATION
        );
        return 0;
    } catch (const std::wstring& error) {
        MessageBoxW(nullptr, error.c_str(), L"Install failed", MB_OK | MB_ICONERROR);
        return 1;
    }
}
