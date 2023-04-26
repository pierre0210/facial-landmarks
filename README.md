# facial-landmarks

## Webcam
Logitech C310 HD
Diagonal FOV 60

## UART Message format

- width (px): width of the mouth bounding box
- height (px): height of the mouth bounding box
- h_angle (degree): horizontal degree to the center of the target
- v_angle (degree): verticle degree to the center of the target

- End of line: CRLF
- Message length: 15 characters

```
width height h_angle v_angle\r\n
```