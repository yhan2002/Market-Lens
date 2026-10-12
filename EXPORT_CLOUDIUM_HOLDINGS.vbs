Option Explicit

If WScript.Arguments.Count <> 2 Then
    WScript.Echo "Usage: EXPORT_CLOUDIUM_HOLDINGS.vbs source.xlsx target.xlsx"
    WScript.Quit 2
End If

Dim sourcePath, targetPath, tempPath, fso, excel, workbook, errorMessage
sourcePath = WScript.Arguments(0)
targetPath = WScript.Arguments(1)
tempPath = targetPath & ".new.xlsx"
Set fso = CreateObject("Scripting.FileSystemObject")

On Error Resume Next
If fso.FileExists(tempPath) Then fso.DeleteFile tempPath, True

Set excel = CreateObject("Excel.Application")
If Err.Number <> 0 Then
    WScript.Echo "Cloudium Excel bridge unavailable."
    WScript.Quit 1
End If

excel.Visible = False
excel.DisplayAlerts = False
excel.AskToUpdateLinks = False
excel.AutomationSecurity = 3
Err.Clear
Set workbook = excel.Workbooks.Open(sourcePath, 0, True)
If Err.Number = 0 Then
    Err.Clear
    workbook.SaveCopyAs tempPath
End If
If Err.Number <> 0 Then errorMessage = Err.Description

If Not workbook Is Nothing Then workbook.Close False
excel.Quit
Set workbook = Nothing
Set excel = Nothing

If Len(errorMessage) > 0 Or Not fso.FileExists(tempPath) Then
    If fso.FileExists(tempPath) Then fso.DeleteFile tempPath, True
    WScript.Echo "Cloudium Excel bridge failed; previous private cache retained."
    WScript.Quit 1
End If

If fso.FileExists(targetPath) Then fso.DeleteFile targetPath, True
fso.MoveFile tempPath, targetPath
WScript.Echo "Cloudium Excel private cache refreshed."
WScript.Quit 0
